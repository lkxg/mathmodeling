"""Package the further architecture experiments alongside the selected result."""
from __future__ import annotations

from .core import OUT, ROOT
from .package_improved import main as package


def main():
    required = ("dual_query.pt", "enhance_balance.pt", "dual_query_distilled.pt",
                "new_architecture_comparison.csv", "new_architecture_robustness.csv",
                "new_architecture_selection.json", "new_architecture_robustness_summary.json",
                "kernel_baseline_summary.json", "kernel_baseline.joblib")
    for name in required:
        path = OUT / "experiments" / name
        if not path.is_file():
            raise FileNotFoundError(path)
    package(ROOT / "outputs/E_q2_q3_new_architectures.zip")


if __name__ == "__main__":
    main()
