"""Exact (Clopper-Pearson) 95% intervals for the image-level rates quoted in the report."""
import sys

from scipy.stats import beta


def cp(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


if __name__ == "__main__":
    pairs = [tuple(map(int, p.split("/"))) for p in sys.argv[1:]] or [(29, 29), (23, 29), (28, 29), (24, 29), (0, 40), (4, 40), (1, 40)]
    for k, n in pairs:
        lo, hi = cp(k, n)
        print(f"{k}/{n} = {k / n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
