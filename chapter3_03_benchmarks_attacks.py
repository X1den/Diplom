from __future__ import annotations

import statistics
import time

from chapter3_01_bigint_primality import (
    BigInt32,
    factor_semiprime,
    generate_prime,
    lucas_lehmer,
    miller_rabin,
    sliding_window_mod_pow,
)

from chapter3_02_public_key_protocols import (
    elgamal_keygen,
    elgamal_sign,
    elgamal_verify,
    generate_wiener_vulnerable_rsa,
    rabin_decrypt,
    rabin_encrypt,
    rabin_keygen,
    recover_elgamal_secret_from_known_nonce,
    recover_rsa_private_key_by_factorization,
    rsa_decrypt,
    rsa_encrypt,
    rsa_keygen,
    rsa_sign,
    rsa_verify,
    wiener_attack,
)

def bool_text(value: bool) -> str:
    return "да" if value else "нет"

def measure(label: str, func, repeats: int = 5):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = func()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        samples.append(elapsed_ms)

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples)
    print(f"{label:<48} среднее={mean:9.3f} мс  отклонение={stdev:8.3f} мс")
    return result

def benchmark_bigint_arithmetic() -> None:
    print("\n=== Арифметика 32-битных блоков ===")
    a = BigInt32.from_int((1 << 4096) + 123456789)
    b = BigInt32.from_int((1 << 4095) + 987654321)

    school = measure("школьное умножение 4096 бит", lambda: a.school_mul(b), repeats=3)
    kara = measure("умножение Карацубы 4096 бит", lambda: a.karatsuba_mul(b), repeats=3)
    assert school.to_int() == kara.to_int() == a.to_int() * b.to_int()

    modulus = generate_prime(256)
    base = 1234567890123456789012345678901234567890
    exponent = (1 << 256) + 12345
    result = measure(
        "модульное возведение скользящим окном",
        lambda: sliding_window_mod_pow(base, exponent, modulus, window=5),
        repeats=5,
    )
    assert result == pow(base, exponent, modulus)

def benchmark_primality_tests() -> None:
    print("\n=== Проверки простоты ===")
    prime64 = 18446744073709551557
    composite64 = prime64 * 101
    print(f"Миллер-Рабин, простое 64 бит    = {bool_text(miller_rabin(prime64))}")
    print(f"Миллер-Рабин, составное 64 бит  = {bool_text(miller_rabin(composite64))}")
    print(f"Люка-Лемер, p=31                = {bool_text(lucas_lehmer(31))}")
    measure("генерация простого 128 бит", lambda: generate_prime(128), repeats=3)

def benchmark_protocols() -> None:
    print("\n=== Протоколы с открытым ключом ===")
    rsa_key = measure("RSA-512: генерация ключа", lambda: rsa_keygen(512), repeats=3)
    rsa_message = "сообщение для RSA".encode("utf-8")
    rsa_ciphertext = measure("RSA-512: шифрование", lambda: rsa_encrypt(rsa_message, rsa_key), repeats=10)
    decrypted = measure("RSA-512: расшифрование с КТО", lambda: rsa_decrypt(rsa_ciphertext, rsa_key), repeats=10)
    assert decrypted == rsa_message
    signature = measure("RSA: подпись", lambda: rsa_sign(rsa_message, rsa_key), repeats=5)
    assert rsa_verify(rsa_message, signature, rsa_key)

    rabin_key = measure("Рабин-512: генерация ключа", lambda: rabin_keygen(512), repeats=3)
    rabin_message = "сообщение Рабина".encode("utf-8")
    rabin_ciphertext = measure("Рабин: шифрование", lambda: rabin_encrypt(rabin_message, rabin_key), repeats=10)
    assert measure("Рабин: расшифрование", lambda: rabin_decrypt(rabin_ciphertext, rabin_key), repeats=10) == rabin_message

    elgamal_key = measure("Эль-Гамаль-128: генерация ключа", lambda: elgamal_keygen(128), repeats=3)
    eg_message = "сообщение Эль-Гамаля".encode("utf-8")
    r, s, _ = measure("Эль-Гамаль: подпись", lambda: elgamal_sign(eg_message, elgamal_key), repeats=5)
    assert elgamal_verify(eg_message, (r, s), elgamal_key)

def simulate_attacks() -> None:
    print("\n=== Учебные атаки ===")
    weak_rsa = rsa_keygen(80)
    p, q, d = measure(
        "ро-метод Полларда: факторизация RSA",
        lambda: recover_rsa_private_key_by_factorization(weak_rsa.n, weak_rsa.e),
        repeats=1,
    )
    print(f"модуль разложен: p={p}, q={q}, d восстановлен={bool_text(d == weak_rsa.d)}")

    vulnerable = measure(
        "генерация RSA, уязвимого к Винеру",
        lambda: generate_wiener_vulnerable_rsa(96),
        repeats=1,
    )
    recovered_d = measure("атака Винера", lambda: wiener_attack(vulnerable.n, vulnerable.e), repeats=1)
    print(f"атака Винера восстановила d = {bool_text(recovered_d == vulnerable.d)}")

    semiprime = 1000003 * 1009837
    factor_pair = measure("ро-метод Полларда: полупростое", lambda: factor_semiprime(semiprime), repeats=3)
    print(f"{semiprime} = {factor_pair[0]} * {factor_pair[1]}")

    key = elgamal_keygen(128)
    message = "известный k".encode("utf-8")
    r, s, k = elgamal_sign(message, key)
    recovered_x = recover_elgamal_secret_from_known_nonce(message, (r, s), key, k)
    print(f"Эль-Гамаль: восстановление при известном k = {bool_text(recovered_x == key.x)}")

def main() -> None:
    benchmark_bigint_arithmetic()
    benchmark_primality_tests()
    benchmark_protocols()
    simulate_attacks()

if __name__ == "__main__":
    main()
