"""Validate selected-model specialty outputs and explanation identities."""
from __future__ import annotations

import csv
import json

from .core import OUT


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    root=OUT/"experiments"
    q2=read_csv(root/"attachment3_predictions_selected.csv")
    q3=read_csv(root/"attachment4_predictions_explanations_selected.csv")
    timing=read_csv(root/"attachment4_evidence_times_selected.csv")
    details=json.loads((root/"attachment4_explanations_selected.json").read_text())
    audit=json.loads((root/"attachment4_input_audit_selected.json").read_text())
    assert len(q2)==len({x["sample_id"] for x in q2})==30
    assert len(q3)==len(details)==len(audit)==len({x["sample_id"] for x in q3})==20
    for row in q2+q3:
        strength=float(row["intensity"])
        assert -3<=strength<=3
        assert ((row["polarity"]=="Neutral" and strength==0) or
                (row["polarity"]=="Negative" and strength<0) or
                (row["polarity"]=="Positive" and strength>0))
    for row in q2:
        assert abs(sum(float(row[f"prob_{n}"]) for n in ("negative","neutral","positive"))-1)<2e-5
    for item in details:
        target=item["predicted_class"]
        assert abs(sum(item["modality_contribution"].values())-
                   (item["class_logits"][target]-item["baseline_class_logit"]))<2e-5
        assert abs(sum(item["modality_share"].values())-1)<2e-5
        row=next(x for x in q3 if x["sample_id"]==item["sample_id"])
        assert abs(float(row["intensity"])-item["reported_intensity"])<1e-6
    assert len(timing)==sum(len(item["windows"]) for item in details)
    assert sum(r["time_status"]=="aligned" for r in timing)==173
    assert [x["sample_id"] for x in audit if not x["modality_available"]["vision"]]==["13"]
    print(f"PASS: {len(q2)} attachment-3 predictions, {len(q3)} attachment-4 explanations, {len(timing)} evidence windows")


if __name__=="__main__":
    main()
