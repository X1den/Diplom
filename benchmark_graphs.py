from __future__ import annotations

import csv
import secrets
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chapter3_01_bigint_primality import (
    BigInt32,
    factor_semiprime,
    generate_prime,
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
    rsa_decrypt,
    rsa_encrypt,
    rsa_keygen,
    rsa_sign,
    rsa_verify,
    wiener_attack,
)


# --------------------------------------------------
# Общие настройки
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "benchmark_results.csv"
RUNS_PER_POINT = 100

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False

BenchmarkRow = dict[str, str | int | float]
BenchmarkFunc = Callable[[], Any]


def measure_ms(
    func: BenchmarkFunc,
    runs: int = RUNS_PER_POINT,
) -> tuple[float, float, Any]:
    """Возвращает среднее время, стандартное отклонение и последний результат."""
    samples: list[float] = []
    result: Any = None

    for _ in range(runs):
        start = time.perf_counter_ns()
        result = func()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        samples.append(elapsed_ms)

    mean_ms = statistics.mean(samples)
    stdev_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean_ms, stdev_ms, result


def format_ms(value: float) -> str:
    """Форматирует миллисекунды без потери малых ненулевых значений."""
    if value == 0:
        return "0"
    if value < 0.001:
        return f"{value:.2e}"
    if value < 0.01:
        return f"{value:.5f}"
    if value < 0.1:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def add_result(
    rows: list[BenchmarkRow],
    graph: str,
    series: str,
    size_bits: int,
    mean_ms: float,
    stdev_ms: float,
) -> None:
    rows.append(
        {
            "graph": graph,
            "series": series,
            "size_bits": size_bits,
            "runs": RUNS_PER_POINT,
            "mean_ms": mean_ms,
            "stdev_ms": stdev_ms,
        }
    )


def save_plot(filename: str) -> Path:
    path = SCRIPT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def add_bar_labels(bars) -> None:
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            format_ms(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_grouped_bars(
    x_labels: list[int],
    series: dict[str, list[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    log_scale: bool = False,
) -> Path:
    positions = list(range(len(x_labels)))
    width = min(0.8 / len(series), 0.22)
    offset_start = -width * (len(series) - 1) / 2

    plt.figure()
    for index, (label, values) in enumerate(series.items()):
        offsets = [x + offset_start + index * width for x in positions]
        bars = plt.bar(offsets, values, width, label=label)
        add_bar_labels(bars)

    plt.xticks(positions, x_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    return save_plot(filename)


def write_results(rows: list[BenchmarkRow]) -> Path:
    fieldnames = [
        "graph",
        "series",
        "size_bits",
        "runs",
        "mean_ms",
        "stdev_ms",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return RESULTS_CSV


# --------------------------------------------------
# 1. Карацуба против школьного умножения
# --------------------------------------------------

def benchmark_multiplication(rows: list[BenchmarkRow]) -> Path:
    bit_sizes = [512, 1024, 2048, 4096, 8192]
    school_times: list[float] = []
    karatsuba_times: list[float] = []

    print(f"Тест умножения: {RUNS_PER_POINT} запусков на точку...")

    for bits in bit_sizes:
        a = BigInt32.from_int(secrets.randbits(bits))
        b = BigInt32.from_int(secrets.randbits(bits))

        school_result = a.school_mul(b)
        karatsuba_result = a.karatsuba_mul(b)
        assert school_result.to_int() == karatsuba_result.to_int() == a.to_int() * b.to_int()

        school_time, school_stdev, _ = measure_ms(lambda: a.school_mul(b))
        karatsuba_time, karatsuba_stdev, _ = measure_ms(lambda: a.karatsuba_mul(b))

        school_times.append(school_time)
        karatsuba_times.append(karatsuba_time)
        add_result(rows, "multiplication", "school", bits, school_time, school_stdev)
        add_result(rows, "multiplication", "karatsuba", bits, karatsuba_time, karatsuba_stdev)
        print(f"{bits} бит завершено")

    plt.figure()
    plt.plot(bit_sizes, school_times, marker="o", label="Школьное умножение")
    plt.plot(bit_sizes, karatsuba_times, marker="s", label="Карацуба")
    plt.xlabel("Размер числа (бит)")
    plt.ylabel("Среднее время (мс)")
    plt.title("Сравнение алгоритмов умножения")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return save_plot("graph_multiplication.png")


# --------------------------------------------------
# 2. Генерация простых чисел
# --------------------------------------------------

def benchmark_primes(rows: list[BenchmarkRow]) -> Path:
    bit_sizes = [128, 256, 512, 1024]
    times: list[float] = []

    print(f"Тест генерации простых чисел: {RUNS_PER_POINT} запусков на точку...")

    for bits in bit_sizes:
        elapsed, stdev, _ = measure_ms(lambda: generate_prime(bits))
        times.append(elapsed)
        add_result(rows, "prime_generation", "generate_prime", bits, elapsed, stdev)
        print(f"{bits} бит завершено")

    plt.figure()
    plt.plot(bit_sizes, times, marker="o")
    plt.xlabel("Размер простого числа (бит)")
    plt.ylabel("Среднее время (мс)")
    plt.title("Время генерации простых чисел")
    plt.grid(True, linestyle="--", alpha=0.6)
    return save_plot("graph_prime_generation.png")


# --------------------------------------------------
# 3. Модульное возведение в степень
# --------------------------------------------------

def benchmark_modular_exponentiation(rows: list[BenchmarkRow]) -> Path:
    bit_sizes = [128, 256, 512, 1024]
    builtin_times: list[float] = []
    sliding_window_times: list[float] = []

    print(f"Тест модульного возведения в степень: {RUNS_PER_POINT} запусков на точку...")

    for bits in bit_sizes:
        modulus = generate_prime(bits)
        base = secrets.randbelow(modulus - 2) + 2
        exponent = secrets.randbits(bits)

        expected = pow(base, exponent, modulus)
        actual = sliding_window_mod_pow(base, exponent, modulus, window=5)
        assert actual == expected

        builtin_time, builtin_stdev, _ = measure_ms(lambda: pow(base, exponent, modulus))
        sliding_time, sliding_stdev, _ = measure_ms(
            lambda: sliding_window_mod_pow(base, exponent, modulus, window=5)
        )

        builtin_times.append(builtin_time)
        sliding_window_times.append(sliding_time)
        add_result(rows, "modular_exponentiation", "pow", bits, builtin_time, builtin_stdev)
        add_result(
            rows,
            "modular_exponentiation",
            "sliding_window",
            bits,
            sliding_time,
            sliding_stdev,
        )
        print(f"{bits} бит завершено")

    plt.figure()
    plt.plot(bit_sizes, builtin_times, marker="o", label="Встроенный pow")
    plt.plot(bit_sizes, sliding_window_times, marker="s", label="Скользящее окно")
    plt.xlabel("Размер модуля (бит)")
    plt.ylabel("Среднее время (мс)")
    plt.title("Модульное возведение в степень")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return save_plot("graph_modular_exponentiation.png")


# --------------------------------------------------
# 4. RSA
# --------------------------------------------------

def benchmark_rsa(rows: list[BenchmarkRow]) -> Path:
    key_sizes = [256, 512, 1024]
    keygen_times: list[float] = []
    encrypt_times: list[float] = []
    decrypt_times: list[float] = []
    sign_times: list[float] = []
    verify_times: list[float] = []

    print(f"Тест RSA: {RUNS_PER_POINT} запусков на точку...")

    message = b"RSA graph message"

    for bits in key_sizes:
        keygen_time, keygen_stdev, rsa_key = measure_ms(lambda: rsa_keygen(bits))

        cipher = rsa_encrypt(message, rsa_key)
        assert rsa_decrypt(cipher, rsa_key) == message
        signature = rsa_sign(message, rsa_key)
        assert rsa_verify(message, signature, rsa_key)

        encrypt_time, encrypt_stdev, _ = measure_ms(lambda: rsa_encrypt(message, rsa_key))
        decrypt_time, decrypt_stdev, _ = measure_ms(lambda: rsa_decrypt(cipher, rsa_key))
        sign_time, sign_stdev, _ = measure_ms(lambda: rsa_sign(message, rsa_key))
        verify_time, verify_stdev, _ = measure_ms(lambda: rsa_verify(message, signature, rsa_key))

        keygen_times.append(keygen_time)
        encrypt_times.append(encrypt_time)
        decrypt_times.append(decrypt_time)
        sign_times.append(sign_time)
        verify_times.append(verify_time)

        for series, mean_ms, stdev_ms in [
            ("keygen", keygen_time, keygen_stdev),
            ("encrypt", encrypt_time, encrypt_stdev),
            ("decrypt", decrypt_time, decrypt_stdev),
            ("sign", sign_time, sign_stdev),
            ("verify", verify_time, verify_stdev),
        ]:
            add_result(rows, "rsa", series, bits, mean_ms, stdev_ms)

        print(f"RSA {bits} бит завершено")

    return plot_grouped_bars(
        key_sizes,
        {
            "Генерация": keygen_times,
            "Шифрование": encrypt_times,
            "Дешифрование": decrypt_times,
            "Подпись": sign_times,
            "Проверка": verify_times,
        },
        xlabel="Размер ключа (бит)",
        ylabel="Среднее время (мс, лог. шкала)",
        title="Производительность RSA",
        filename="graph_rsa.png",
        log_scale=True,
    )


# --------------------------------------------------
# 5. Rabin
# --------------------------------------------------

def benchmark_rabin(rows: list[BenchmarkRow]) -> Path:
    key_sizes = [256, 512, 1024]
    keygen_times: list[float] = []
    encrypt_times: list[float] = []
    decrypt_times: list[float] = []

    print(f"Тест Rabin: {RUNS_PER_POINT} запусков на точку...")

    message = b"Rabin graph"

    for bits in key_sizes:
        keygen_time, keygen_stdev, key = measure_ms(lambda: rabin_keygen(bits))

        cipher = rabin_encrypt(message, key)
        assert rabin_decrypt(cipher, key) == message

        encrypt_time, encrypt_stdev, _ = measure_ms(lambda: rabin_encrypt(message, key))
        decrypt_time, decrypt_stdev, _ = measure_ms(lambda: rabin_decrypt(cipher, key))

        keygen_times.append(keygen_time)
        encrypt_times.append(encrypt_time)
        decrypt_times.append(decrypt_time)

        for series, mean_ms, stdev_ms in [
            ("keygen", keygen_time, keygen_stdev),
            ("encrypt", encrypt_time, encrypt_stdev),
            ("decrypt", decrypt_time, decrypt_stdev),
        ]:
            add_result(rows, "rabin", series, bits, mean_ms, stdev_ms)

        print(f"Rabin {bits} бит завершено")

    return plot_grouped_bars(
        key_sizes,
        {
            "Генерация": keygen_times,
            "Шифрование": encrypt_times,
            "Дешифрование": decrypt_times,
        },
        xlabel="Размер ключа (бит)",
        ylabel="Среднее время (мс, лог. шкала)",
        title="Производительность криптосистемы Rabin",
        filename="graph_rabin.png",
        log_scale=True,
    )


# --------------------------------------------------
# 6. Подпись Эль-Гамаля
# --------------------------------------------------

def benchmark_elgamal(rows: list[BenchmarkRow]) -> Path:
    key_sizes = [96, 128, 160, 192]
    keygen_times: list[float] = []
    sign_times: list[float] = []
    verify_times: list[float] = []

    print(f"Тест подписи Эль-Гамаля: {RUNS_PER_POINT} запусков на точку...")

    message = b"ElGamal graph message"

    for bits in key_sizes:
        keygen_time, keygen_stdev, key = measure_ms(lambda: elgamal_keygen(bits))

        r, s, _ = elgamal_sign(message, key)
        assert elgamal_verify(message, (r, s), key)

        sign_time, sign_stdev, _ = measure_ms(lambda: elgamal_sign(message, key))
        verify_time, verify_stdev, _ = measure_ms(lambda: elgamal_verify(message, (r, s), key))

        keygen_times.append(keygen_time)
        sign_times.append(sign_time)
        verify_times.append(verify_time)

        for series, mean_ms, stdev_ms in [
            ("keygen", keygen_time, keygen_stdev),
            ("sign", sign_time, sign_stdev),
            ("verify", verify_time, verify_stdev),
        ]:
            add_result(rows, "elgamal_signature", series, bits, mean_ms, stdev_ms)

        print(f"ElGamal {bits} бит завершено")

    return plot_grouped_bars(
        key_sizes,
        {
            "Генерация": keygen_times,
            "Подпись": sign_times,
            "Проверка": verify_times,
        },
        xlabel="Размер ключа (бит)",
        ylabel="Среднее время (мс, лог. шкала)",
        title="Производительность подписи Эль-Гамаля",
        filename="graph_elgamal_signature.png",
        log_scale=True,
    )


# --------------------------------------------------
# 7. Учебные атаки
# --------------------------------------------------

def benchmark_attacks(rows: list[BenchmarkRow]) -> Path:
    rsa_sizes = [48, 56, 64, 72]
    wiener_sizes = [64, 80, 96, 112]
    elgamal_sizes = [64, 96, 128, 160]

    factor_times: list[float] = []
    wiener_times: list[float] = []
    nonce_times: list[float] = []

    print(f"Тест учебных атак: {RUNS_PER_POINT} запусков на точку...")

    for bits in rsa_sizes:
        p = generate_prime(bits // 2)
        q = generate_prime(bits - bits // 2)
        semiprime = p * q
        factor_time, factor_stdev, recovered = measure_ms(
            lambda: factor_semiprime(semiprime)
        )
        assert recovered[0] * recovered[1] == semiprime

        factor_times.append(factor_time)
        add_result(
            rows,
            "attacks",
            "pollard_rho_factorization",
            bits,
            factor_time,
            factor_stdev,
        )
        print(f"Факторизация {bits} бит завершена")

    for bits in wiener_sizes:
        vulnerable = generate_wiener_vulnerable_rsa(bits)
        wiener_time, wiener_stdev, recovered_d = measure_ms(
            lambda: wiener_attack(vulnerable.n, vulnerable.e)
        )
        assert recovered_d == vulnerable.d

        wiener_times.append(wiener_time)
        add_result(rows, "attacks", "wiener_attack", bits, wiener_time, wiener_stdev)
        print(f"Атака Винера {bits} бит завершена")

    for bits in elgamal_sizes:
        key = elgamal_keygen(bits)
        message = b"nonce disclosure"
        r, s, k = elgamal_sign(message, key)
        nonce_time, nonce_stdev, recovered_x = measure_ms(
            lambda: recover_elgamal_secret_from_known_nonce(message, (r, s), key, k)
        )
        assert recovered_x == key.x

        nonce_times.append(nonce_time)
        add_result(
            rows,
            "attacks",
            "elgamal_known_nonce",
            bits,
            nonce_time,
            nonce_stdev,
        )
        print(f"Восстановление nonce Эль-Гамаля {bits} бит завершено")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Время выполнения учебных атак", fontsize=15)

    attack_plots = [
        (axes[0], rsa_sizes, factor_times, "Pollard rho", "Модуль RSA (бит)", "o"),
        (axes[1], wiener_sizes, wiener_times, "Атака Винера", "Модуль RSA (бит)", "s"),
        (
            axes[2],
            elgamal_sizes,
            nonce_times,
            "Known nonce Эль-Гамаля",
            "Ключ Эль-Гамаля (бит)",
            "^",
        ),
    ]

    for axis, sizes, times, title, xlabel, marker in attack_plots:
        axis.plot(sizes, times, marker=marker)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Среднее время (мс)")
        axis.grid(True, linestyle="--", alpha=0.6)

        for size, elapsed in zip(sizes, times):
            axis.annotate(
                format_ms(elapsed),
                xy=(size, elapsed),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    return save_plot("graph_attacks.png")


# --------------------------------------------------
# Главная функция
# --------------------------------------------------

def main() -> None:
    rows: list[BenchmarkRow] = []

    graph_paths = [
        benchmark_multiplication(rows),
        benchmark_primes(rows),
        benchmark_modular_exponentiation(rows),
        benchmark_rsa(rows),
        benchmark_rabin(rows),
        benchmark_elgamal(rows),
        benchmark_attacks(rows),
    ]
    results_path = write_results(rows)

    print()
    print("Готово.")
    print(f"Каждая точка рассчитана по {RUNS_PER_POINT} запускам.")
    print("Созданы файлы:")
    for path in graph_paths:
        print(path.name)
    print(results_path.name)


if __name__ == "__main__":
    main()