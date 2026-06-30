from __future__ import annotations

import math
import secrets
from dataclasses import dataclass


WORD_BITS = 32
WORD_BASE = 1 << WORD_BITS
WORD_MASK = WORD_BASE - 1
KARATSUBA_CUTOFF = 24
_rng = secrets.SystemRandom()


def _strip(limbs: list[int]) -> list[int]:
    while limbs and limbs[-1] == 0:
        limbs.pop()
    return limbs


def _cmp_abs(a: list[int], b: list[int]) -> int:
    a = _strip(a[:])
    b = _strip(b[:])
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            return -1 if x < y else 1
    return 0


def _add_abs(a: list[int], b: list[int]) -> list[int]:
    n = max(len(a), len(b))
    result: list[int] = []
    carry = 0

    for i in range(n):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        total = x + y + carry
        result.append(total & WORD_MASK)
        carry = total >> WORD_BITS

    if carry:
        result.append(carry)
    return _strip(result)


def _sub_abs(a: list[int], b: list[int]) -> list[int]:
    """Return |a| - |b| for |a| >= |b|."""
    result: list[int] = []
    borrow = 0

    for i in range(len(a)):
        x = a[i]
        y = b[i] if i < len(b) else 0
        value = x - y - borrow
        if value < 0:
            value += WORD_BASE
            borrow = 1
        else:
            borrow = 0
        result.append(value)

    return _strip(result)


def _shift_limbs(a: list[int], count: int) -> list[int]:
    if not a:
        return []
    return [0] * count + a


def _school_mul_abs(a: list[int], b: list[int]) -> list[int]:
    if not a or not b:
        return []

    result = [0] * (len(a) + len(b))
    for i, x in enumerate(a):
        carry = 0
        for j, y in enumerate(b):
            total = result[i + j] + x * y + carry
            result[i + j] = total & WORD_MASK
            carry = total >> WORD_BITS
        result[i + len(b)] += carry

    return _strip(result)


def _karatsuba_mul_abs(a: list[int], b: list[int]) -> list[int]:
    a = _strip(a[:])
    b = _strip(b[:])
    if not a or not b:
        return []
    if min(len(a), len(b)) <= KARATSUBA_CUTOFF:
        return _school_mul_abs(a, b)

    n = max(len(a), len(b))
    m = n // 2

    a0, a1 = a[:m], a[m:]
    b0, b1 = b[:m], b[m:]

    z0 = _karatsuba_mul_abs(a0, b0)
    z2 = _karatsuba_mul_abs(a1, b1)
    z1 = _karatsuba_mul_abs(_add_abs(a0, a1), _add_abs(b0, b1))
    z1 = _sub_abs(_sub_abs(z1, z2), z0)

    result = _add_abs(z0, _shift_limbs(z1, m))
    result = _add_abs(result, _shift_limbs(z2, 2 * m))
    return _strip(result)


@dataclass
class BigInt32:
    """Signed integer stored as little-endian 32-bit limbs."""

    sign: int
    limbs: list[int]

    def __post_init__(self) -> None:
        self.limbs = [x & WORD_MASK for x in self.limbs]
        _strip(self.limbs)
        if not self.limbs:
            self.sign = 0
        else:
            self.sign = -1 if self.sign < 0 else 1

    @classmethod
    def from_int(cls, value: int) -> "BigInt32":
        if value == 0:
            return cls(0, [])

        sign = -1 if value < 0 else 1
        value = abs(value)
        limbs: list[int] = []

        while value:
            limbs.append(value & WORD_MASK)
            value >>= WORD_BITS

        return cls(sign, limbs)

    def to_int(self) -> int:
        value = 0
        for limb in reversed(self.limbs):
            value = (value << WORD_BITS) | limb
        return value if self.sign >= 0 else -value

    def __add__(self, other: "BigInt32") -> "BigInt32":
        if self.sign == 0:
            return BigInt32(other.sign, other.limbs[:])
        if other.sign == 0:
            return BigInt32(self.sign, self.limbs[:])
        if self.sign == other.sign:
            return BigInt32(self.sign, _add_abs(self.limbs, other.limbs))

        cmp_result = _cmp_abs(self.limbs, other.limbs)
        if cmp_result == 0:
            return BigInt32(0, [])
        if cmp_result > 0:
            return BigInt32(self.sign, _sub_abs(self.limbs, other.limbs))
        return BigInt32(other.sign, _sub_abs(other.limbs, self.limbs))

    def __neg__(self) -> "BigInt32":
        return BigInt32(-self.sign, self.limbs[:])

    def __sub__(self, other: "BigInt32") -> "BigInt32":
        return self + (-other)

    def school_mul(self, other: "BigInt32") -> "BigInt32":
        sign = self.sign * other.sign
        return BigInt32(sign, _school_mul_abs(self.limbs, other.limbs))

    def karatsuba_mul(self, other: "BigInt32") -> "BigInt32":
        sign = self.sign * other.sign
        return BigInt32(sign, _karatsuba_mul_abs(self.limbs, other.limbs))

    def __mul__(self, other: "BigInt32") -> "BigInt32":
        return self.karatsuba_mul(other)

    def __repr__(self) -> str:
        return f"BigInt32({self.to_int()})"


def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1

    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y

    return old_r, old_x, old_y


def mod_inverse(a: int, modulus: int) -> int:
    g, x, _ = extended_gcd(a % modulus, modulus)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % modulus


@dataclass(frozen=True)
class BarrettContext:
    modulus: int
    k: int
    mu: int

    @classmethod
    def create(cls, modulus: int) -> "BarrettContext":
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        k = max(1, (modulus.bit_length() + WORD_BITS - 1) // WORD_BITS)
        mu = (WORD_BASE ** (2 * k)) // modulus
        return cls(modulus=modulus, k=k, mu=mu)


def barrett_reduce(x: int, ctx: BarrettContext) -> int:
    if x < 0:
        return (-barrett_reduce(-x, ctx)) % ctx.modulus
    if x < ctx.modulus:
        return x

    b_pow = WORD_BASE ** (ctx.k + 1)
    q1 = x // (WORD_BASE ** (ctx.k - 1))
    q2 = q1 * ctx.mu
    q3 = q2 // (WORD_BASE ** (ctx.k + 1))
    r1 = x % b_pow
    r2 = (q3 * ctx.modulus) % b_pow
    r = r1 - r2

    if r < 0:
        r += b_pow
    while r >= ctx.modulus:
        r -= ctx.modulus
    return r


def barrett_mul_mod(a: int, b: int, ctx: BarrettContext) -> int:
    return barrett_reduce((a % ctx.modulus) * (b % ctx.modulus), ctx)


@dataclass(frozen=True)
class MontgomeryContext:
    modulus: int
    r_bits: int
    r: int
    r_mask: int
    n_prime: int

    @classmethod
    def create(cls, modulus: int) -> "MontgomeryContext":
        if modulus <= 0 or modulus % 2 == 0:
            raise ValueError("Montgomery reduction requires an odd modulus")
        words = max(1, (modulus.bit_length() + WORD_BITS - 1) // WORD_BITS)
        r_bits = words * WORD_BITS
        r = 1 << r_bits
        n_prime = (-mod_inverse(modulus, r)) & (r - 1)
        return cls(modulus=modulus, r_bits=r_bits, r=r, r_mask=r - 1, n_prime=n_prime)

    def reduce(self, t: int) -> int:
        m = ((t & self.r_mask) * self.n_prime) & self.r_mask
        u = (t + m * self.modulus) >> self.r_bits
        while u >= self.modulus:
            u -= self.modulus
        return u

    def to_montgomery(self, value: int) -> int:
        return (value % self.modulus) * self.r % self.modulus

    def from_montgomery(self, value: int) -> int:
        return self.reduce(value)

    def multiply(self, a: int, b: int) -> int:
        return self.reduce(a * b)


def montgomery_pow(base: int, exponent: int, modulus: int) -> int:
    ctx = MontgomeryContext.create(modulus)
    result = ctx.to_montgomery(1)
    x = ctx.to_montgomery(base)

    while exponent:
        if exponent & 1:
            result = ctx.multiply(result, x)
        x = ctx.multiply(x, x)
        exponent >>= 1

    return ctx.from_montgomery(result)


def sliding_window_mod_pow(base: int, exponent: int, modulus: int, window: int = 5) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if exponent < 0:
        raise ValueError("negative exponent is not supported")
    if exponent == 0:
        return 1 % modulus

    window = max(1, min(window, 8))
    ctx = BarrettContext.create(modulus)

    def mul(a: int, b: int) -> int:
        return barrett_mul_mod(a, b, ctx)

    base %= modulus
    table = {1: base}
    base_squared = mul(base, base)
    for e in range(3, 1 << window, 2):
        table[e] = mul(table[e - 2], base_squared)

    bits = bin(exponent)[2:]
    result = 1 % modulus
    i = 0

    while i < len(bits):
        if bits[i] == "0":
            result = mul(result, result)
            i += 1
            continue

        j = min(i + window, len(bits))
        while bits[j - 1] == "0":
            j -= 1
        value = int(bits[i:j], 2)

        for _ in range(j - i):
            result = mul(result, result)
        result = mul(result, table[value])
        i = j

    return result


_SMALL_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61,
)
_MR64_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MR64_FAST_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _miller_rabin_with_bases(n: int, bases: tuple[int, ...]) -> bool:
    if n < 2:
        return False
    if n in _SMALL_PRIMES:
        return True
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for a in bases:
        a %= n
        if a in (0, 1):
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def miller_rabin(n: int, rounds: int = 32) -> bool:
    """Deterministic for n < 2^64, probabilistic above this bound."""
    if n < (1 << 64):
        return _miller_rabin_with_bases(n, _MR64_FAST_BASES)

    if n < 2:
        return False
    if n in _SMALL_PRIMES:
        return True
    if any(n % p == 0 for p in _SMALL_PRIMES):
        return False

    bases = tuple(_rng.randrange(2, n - 2) for _ in range(rounds))
    return _miller_rabin_with_bases(n, bases)


def lucas_lehmer(mersenne_exponent: int) -> bool:
    """Return True if 2^p - 1 is prime for prime p."""
    p = mersenne_exponent
    if p == 2:
        return True
    if not miller_rabin(p):
        return False

    m = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % m
    return s == 0


def random_odd_integer(bits: int) -> int:
    if bits < 2:
        raise ValueError("bit length must be at least 2")
    value = _rng.getrandbits(bits)
    value |= 1
    value |= 1 << (bits - 1)
    return value


def generate_prime(bits: int, rounds: int = 32) -> int:
    while True:
        candidate = random_odd_integer(bits)
        if miller_rabin(candidate, rounds):
            return candidate


def generate_blum_prime(bits: int) -> int:
    while True:
        p = generate_prime(bits)
        if p % 4 == 3:
            return p


def generate_safe_prime(bits: int) -> tuple[int, int]:
    """Return p, q where p = 2q + 1 and p, q are prime."""
    if bits < 4:
        raise ValueError("safe prime size is too small")
    while True:
        q = generate_prime(bits - 1)
        p = 2 * q + 1
        if miller_rabin(p):
            return p, q


def pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if miller_rabin(n):
        return n

    while True:
        c = _rng.randrange(1, n - 1)
        x = _rng.randrange(2, n - 1)
        y = x
        d = 1

        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)

        if d != n:
            return d


def factor_semiprime(n: int) -> tuple[int, int]:
    p = pollard_rho(n)
    q = n // p
    return min(p, q), max(p, q)


def self_test() -> None:
    values = [
        0,
        1,
        -1,
        2**32 - 1,
        2**64 + 12345,
        -(2**96 + 7),
    ]

    for a in values:
        for b in values:
            aa = BigInt32.from_int(a)
            bb = BigInt32.from_int(b)
            assert (aa + bb).to_int() == a + b
            assert (aa - bb).to_int() == a - b
            assert aa.school_mul(bb).to_int() == a * b
            assert aa.karatsuba_mul(bb).to_int() == a * b

    modulus = 1000000007
    assert sliding_window_mod_pow(17, 123456, modulus) == pow(17, 123456, modulus)
    assert montgomery_pow(17, 123456, modulus) == pow(17, 123456, modulus)
    assert miller_rabin(2**61 - 1)
    assert not miller_rabin(2**61 - 3)
    assert lucas_lehmer(31)


if __name__ == "__main__":
    self_test()
    print("Chapter 3.1 self-test passed")

