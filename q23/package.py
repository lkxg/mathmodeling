"""Build a compact, reproducible question-2/3 deliverable archive."""
from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

from .core import OUT, ROOT


def main():
    target = ROOT / "outputs/E_q2_q3_deliverables.zip"
    required = ["model.pt", "model_baseline.pt", "normalization.npz",
                "metrics.json", "metrics_baseline.json", "validation_robustness.csv",
                "validation_robustness_baseline.csv", "analysis.json",
                "attachment3_predictions.csv", "attachment3_input_audit.json",
                "attachment4_predictions_explanations.csv", "attachment4_explanations.json",
                "attachment4_input_audit.json",
                "attachment4_evidence_times.csv", "attachment4_position_to_time.json",
                "attachment4_alignment_audit.json", "attachment4_cards.html",
                "q2_robustness.png", "q3_modality_effects.png",
                "train_history.json", "train_history_baseline.json"]
    for name in required:
        if not (OUT / name).is_file():
            raise FileNotFoundError(OUT / name)
    source = [p for p in (ROOT / "q23").rglob("*") if p.is_file() and
              "__pycache__" not in p.parts and p.suffix in (".py", ".yaml", ".md", ".txt")]
    # Time localization reuses these question-1 utilities and their fixed aligner config.
    source += [p for p in (ROOT / "q1").iterdir() if p.is_file() and
               p.suffix in (".py", ".yaml", ".txt", ".sh")]
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=6) as z:
        for p in source:
            z.write(p, p.relative_to(ROOT))
        for name in required:
            p = OUT / name
            z.write(p, p.relative_to(ROOT))
        for p in (OUT / "keyframes").glob("*.jpg"):
            z.write(p, p.relative_to(ROOT))
    print(target, f"{target.stat().st_size/1024/1024:.2f} MiB")


if __name__ == "__main__":
    main()
