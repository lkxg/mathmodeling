"""Independent structural checks of prediction and explanation exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import OUT


def read_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    q2 = read_csv("attachment3_predictions.csv")
    q3 = read_csv("attachment4_predictions_explanations.csv")
    explanations = json.loads((OUT / "attachment4_explanations.json").read_text())
    audit = json.loads((OUT / "attachment4_input_audit.json").read_text())
    mapping = json.loads((OUT / "attachment4_position_to_time.json").read_text())
    timed = read_csv("attachment4_evidence_times.csv")
    assert len(q2) == len({r["sample_id"] for r in q2}) == 30
    assert len(q3) == len(explanations) == len(mapping) == len(audit) == len({r["sample_id"] for r in q3}) == 20
    for item in audit:
        row = next(r for r in q3 if r["sample_id"] == item["sample_id"])
        assert str(item["modality_available"]["vision"]) == row["vision_available"]
    assert all(m["token_id_match"] for m in mapping)
    assert all(-3 <= float(r["intensity"]) <= 3 for r in q2 + q3)
    assert all((r["polarity"] == "Neutral" and float(r["intensity"]) == 0) or
               (r["polarity"] == "Negative" and float(r["intensity"]) < 0) or
               (r["polarity"] == "Positive" and float(r["intensity"]) > 0) for r in q2)
    assert all(abs(sum(float(r[f"prob_{n}"]) for n in ("negative", "neutral", "positive")) - 1) < 2e-5 for r in q2)
    for item in explanations:
        target = item["predicted_class"]
        total = sum(item["modality_contribution"].values())
        full_minus_empty = item["class_logits"][target] - item["baseline_class_logit"]
        assert abs(total - full_minus_empty) < 2e-5, (item["sample_id"], total, full_minus_empty)
        assert abs(sum(item["modality_share"].values()) - 1) < 2e-5
    assert len(timed) == sum(len(item["windows"]) for item in explanations)
    print(f"PASS: {len(q2)} attachment-3 predictions, {len(q3)} attachment-4 explanations, {len(timed)} evidence windows")


if __name__ == "__main__":
    main()
