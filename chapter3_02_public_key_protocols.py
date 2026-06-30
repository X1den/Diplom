from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from math import isqrt

from chapter3_01_bigint_primality import (
    factor_semiprime,
    generate_blum_prime,
    generate_prime,
    generate_safe_prime,
    gcd,
    miller_rabin,
    mod_inverse,
)


_rng = secrets.SystemRandom()


def int_to_bytes(value: int, length: int | None = None) -> bytes:
    if value < 0:
        raise ValueError("отрицательное число нельзя преобразовать")
    if length is None:
        length = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, "big")


def bytes_to_int(data: bytes) -> int:
    return int.from_bytes(data, "big")


def sha256_int(data: bytes, modulus: int) -> int:
    return bytes_to_int(hashlib.sha256(data).digest()) % modulus


def crt_pair(a: int, p: int, b: int, q: int) -> int:
    n = p * q
    m1 = q * mod_inverse(q, p)
    m2 = p * mod_inverse(p, q)
    return (a * m1 + b * m2) % n


def nonzero_random_bytes(length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        value = _rng.randrange(1, 256)
        result.append(value)
    return bytes(result)


@dataclass(frozen=True)
class RSAKeyPair:
    n: int
    e: int
    d: int
    p: int
    q: int
    dp: int
    dq: int
    q_inv: int

    @property
    def byte_length(self) -> int:
        return (self.n.bit_length() + 7) // 8


def rsa_keygen(bits: int = 512, e: int = 65537) -> RSAKeyPair:
    if bits < 64:
        raise ValueError("для учебной демонстрации используйте не менее 64 бит")

    half = bits // 2
    while True:
        p = generate_prime(half)
        q = generate_prime(bits - half)
        if p == q:
            continue
        phi = (p - 1) * (q - 1)
        if gcd(e, phi) != 1:
            continue
        n = p * q
        d = mod_inverse(e, phi)
        return RSAKeyPair(
            n=n,
            e=e,
            d=d,
            p=p,
            q=q,
            dp=d % (p - 1),
            dq=d % (q - 1),
            q_inv=mod_inverse(q, p),
        )


def pkcs1_v15_pad(message: bytes, modulus_length: int) -> bytes:
    if len(message) > modulus_length - 11:
        raise ValueError("сообщение слишком длинное для модуля RSA")
    ps_len = modulus_length - len(message) - 3
    return b"\x00\x02" + nonzero_random_bytes(ps_len) + b"\x00" + message


def pkcs1_v15_unpad(block: bytes) -> bytes:
    if len(block) < 11 or not block.startswith(b"\x00\x02"):
        raise ValueError("некорректный префикс дополнения")
    sep = block.find(b"\x00", 2)
    if sep < 10:
        raise ValueError("некорректная строка дополнения")
    return block[sep + 1 :]


def rsa_encrypt(message: bytes, key: RSAKeyPair) -> int:
    block = pkcs1_v15_pad(message, key.byte_length)
    return pow(bytes_to_int(block), key.e, key.n)


def rsa_decrypt(ciphertext: int, key: RSAKeyPair) -> bytes:
    m1 = pow(ciphertext % key.p, key.dp, key.p)
    m2 = pow(ciphertext % key.q, key.dq, key.q)
    h = ((m1 - m2) * key.q_inv) % key.p
    message_int = (m2 + h * key.q) % key.n
    block = int_to_bytes(message_int, key.byte_length)
    return pkcs1_v15_unpad(block)


def rsa_sign(message: bytes, key: RSAKeyPair) -> int:
    digest = hashlib.sha256(message).digest()
    h = bytes_to_int(digest) % key.n
    return pow(h, key.d, key.n)


def rsa_verify(message: bytes, signature: int, key: RSAKeyPair) -> bool:
    digest = hashlib.sha256(message).digest()
    h = bytes_to_int(digest) % key.n
    return pow(signature, key.e, key.n) == h


def recover_rsa_private_key_by_factorization(n: int, e: int) -> tuple[int, int, int]:
    p, q = factor_semiprime(n)
    phi = (p - 1) * (q - 1)
    d = mod_inverse(e, phi)
    return p, q, d


@dataclass(frozen=True)
class RabinKeyPair:
    n: int
    p: int
    q: int

    @property
    def byte_length(self) -> int:
        return (self.n.bit_length() + 7) // 8


def rabin_keygen(bits: int = 512) -> RabinKeyPair:
    if bits < 128:
        raise ValueError("для учебной демонстрации используйте не менее 128 бит")
    half = bits // 2
    while True:
        p = generate_blum_prime(half)
        q = generate_blum_prime(bits - half)
        if p != q:
            return RabinKeyPair(n=p * q, p=p, q=q)


def rabin_encode(message: bytes, modulus_length: int) -> int:
    tag_length = 8
    if len(message) > modulus_length - tag_length - 3:
        raise ValueError("сообщение слишком длинное для модуля Рабина")
    body = len(message).to_bytes(2, "big") + message
    tag = hashlib.sha256(body).digest()[:tag_length]
    encoded = b"\x02" + body + tag
    return bytes_to_int(encoded)


def rabin_decode(candidate: int, modulus_length: int) -> bytes | None:
    data = int_to_bytes(candidate, modulus_length).lstrip(b"\x00")
    if len(data) < 11 or data[0] != 2:
        return None

    length = int.from_bytes(data[1:3], "big")
    end = 3 + length
    if end + 8 != len(data):
        return None

    body = data[1:end]
    message = data[3:end]
    tag = data[end:]
    if hashlib.sha256(body).digest()[:8] != tag:
        return None
    return message


def rabin_encrypt(message: bytes, key: RabinKeyPair) -> int:
    m = rabin_encode(message, key.byte_length)
    if m >= key.n:
        raise ValueError("закодированное сообщение не меньше модуля")
    return pow(m, 2, key.n)


def rabin_decrypt(ciphertext: int, key: RabinKeyPair) -> bytes:
    p, q, n = key.p, key.q, key.n
    mp = pow(ciphertext, (p + 1) // 4, p)
    mq = pow(ciphertext, (q + 1) // 4, q)

    roots = {
        crt_pair(mp, p, mq, q),
        crt_pair((-mp) % p, p, mq, q),
        crt_pair(mp, p, (-mq) % q, q),
        crt_pair((-mp) % p, p, (-mq) % q, q),
    }

    decoded = [rabin_decode(root, key.byte_length) for root in roots]
    decoded = [item for item in decoded if item is not None]
    if len(decoded) != 1:
        raise ValueError("не удалось однозначно выбрать корень Рабина")
    return decoded[0]


@dataclass(frozen=True)
class ElGamalKeyPair:
    p: int
    g: int
    x: int
    y: int


def primitive_root_for_safe_prime(p: int, q: int) -> int:
    if p != 2 * q + 1 or not miller_rabin(p) or not miller_rabin(q):
        raise ValueError("ожидалось безопасное простое p = 2q + 1")
    while True:
        g = _rng.randrange(2, p - 2)
        if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
            return g


def elgamal_keygen(bits: int = 256) -> ElGamalKeyPair:
    p, q = generate_safe_prime(bits)
    g = primitive_root_for_safe_prime(p, q)
    x = _rng.randrange(1, p - 1)
    y = pow(g, x, p)
    return ElGamalKeyPair(p=p, g=g, x=x, y=y)


def elgamal_sign(message: bytes, key: ElGamalKeyPair, k: int | None = None) -> tuple[int, int, int]:
    p, g, x = key.p, key.g, key.x
    modulus = p - 1
    h = sha256_int(message, modulus)

    if k is None:
        while True:
            k = _rng.randrange(2, modulus)
            if gcd(k, modulus) == 1:
                break
    elif gcd(k, modulus) != 1:
        raise ValueError("одноразовый параметр k должен быть обратим по модулю p - 1")

    r = pow(g, k, p)
    s = (mod_inverse(k, modulus) * (h - x * r)) % modulus
    return r, s, k


def elgamal_verify(message: bytes, signature: tuple[int, int], key: ElGamalKeyPair) -> bool:
    p, g, y = key.p, key.g, key.y
    r, s = signature
    if not 1 <= r < p or not 0 <= s < p - 1:
        return False

    h = sha256_int(message, p - 1)
    left = pow(g, h, p)
    right = (pow(y, r, p) * pow(r, s, p)) % p
    return left == right


def solve_linear_congruence(a: int, b: int, modulus: int) -> list[int]:
    d = gcd(a, modulus)
    if b % d != 0:
        return []
    a1, b1, m1 = a // d, b // d, modulus // d
    first = (b1 * mod_inverse(a1 % m1, m1)) % m1
    return [(first + i * m1) % modulus for i in range(d)]


def recover_elgamal_secret_from_known_nonce(
    message: bytes,
    signature: tuple[int, int],
    key: ElGamalKeyPair,
    k: int,
) -> int:
    r, s = signature
    modulus = key.p - 1
    h = sha256_int(message, modulus)
    candidates = solve_linear_congruence(r, (h - s * k) % modulus, modulus)
    for candidate in candidates:
        if pow(key.g, candidate, key.p) == key.y:
            return candidate
    raise ValueError("секретный ключ не восстановлен")


def continued_fraction(numerator: int, denominator: int) -> list[int]:
    values: list[int] = []
    while denominator:
        q = numerator // denominator
        values.append(q)
        numerator, denominator = denominator, numerator - q * denominator
    return values


def convergents(cf: list[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    n0, n1 = 0, 1
    d0, d1 = 1, 0
    for q in cf:
        n0, n1 = n1, q * n1 + n0
        d0, d1 = d1, q * d1 + d0
        result.append((n1, d1))
    return result


def wiener_attack(n: int, e: int) -> int | None:
    for k, d in convergents(continued_fraction(e, n)):
        if k == 0:
            continue
        ed_minus_1 = e * d - 1
        if ed_minus_1 % k != 0:
            continue

        phi = ed_minus_1 // k
        s = n - phi + 1
        discriminant = s * s - 4 * n
        if discriminant < 0:
            continue

        root = isqrt(discriminant)
        if root * root != discriminant:
            continue
        if (s + root) % 2 != 0:
            continue

        p = (s + root) // 2
        q = (s - root) // 2
        if p * q == n:
            return d
    return None


def generate_wiener_vulnerable_rsa(bits: int = 96) -> RSAKeyPair:
    half = bits // 2
    while True:
        p = generate_prime(half)
        q = generate_prime(bits - half)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)
        limit = max(3, int(n ** 0.25) // 3)
        d = _rng.randrange(2, limit)
        if gcd(d, phi) != 1:
            continue
        e = mod_inverse(d, phi)
        key = RSAKeyPair(
            n=n,
            e=e,
            d=d,
            p=p,
            q=q,
            dp=d % (p - 1),
            dq=d % (q - 1),
            q_inv=mod_inverse(q, p),
        )
        if wiener_attack(n, e) == d:
            return key


def self_test() -> None:
    rsa_key = rsa_keygen(256)
    message = "тест RSA".encode("utf-8")
    ciphertext = rsa_encrypt(message, rsa_key)
    assert rsa_decrypt(ciphertext, rsa_key) == message
    assert rsa_verify(message, rsa_sign(message, rsa_key), rsa_key)

    rabin_key = rabin_keygen(256)
    rabin_message = "тест Рабина".encode("utf-8")
    assert rabin_decrypt(rabin_encrypt(rabin_message, rabin_key), rabin_key) == rabin_message

    elgamal_key = elgamal_keygen(96)
    elgamal_message = "тест Эль-Гамаля".encode("utf-8")
    r, s, k = elgamal_sign(elgamal_message, elgamal_key)
    assert elgamal_verify(elgamal_message, (r, s), elgamal_key)
    assert recover_elgamal_secret_from_known_nonce(elgamal_message, (r, s), elgamal_key, k) == elgamal_key.x

    weak_key = generate_wiener_vulnerable_rsa(96)
    assert wiener_attack(weak_key.n, weak_key.e) == weak_key.d


if __name__ == "__main__":
    self_test()
    print("Самопроверка главы 3.2 пройдена")
