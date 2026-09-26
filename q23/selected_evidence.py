"""Attach source-text and video times to selected-model explanations."""
from __future__ import annotations

import csv
import json

from .core import OUT, save_csv, save_json


def main():
    root=OUT/"experiments"
    mapping={x["sample_id"]:x for x in json.loads((OUT/"attachment4_position_to_time.json").read_text())}
    explanations=json.loads((root/"attachment4_explanations_selected.json").read_text())
    with (root/"attachment4_predictions_explanations_selected.csv").open(encoding="utf-8-sig",newline="") as f:
        summary=list(csv.DictReader(f))
    rows=[]
    for exp in explanations:
        timing=mapping[exp["sample_id"]]
        for win in exp["windows"]:
            segment=timing["positions"][win["start_position"]:win["end_position_exclusive"]]
            observed=[x for x in segment if x is not None]
            rows.append({"sample_id":exp["sample_id"],"polarity":exp["predicted_label"],
                         "intensity":round(exp["reported_intensity"],6),"main_modality":exp["main_modality"],
                         "evidence_modality":win["modality"],"start_position":win["start_position"],
                         "end_position_exclusive":win["end_position_exclusive"],
                         "target_logit_drop":round(win["target_logit_drop"],6),
                         "token_id_match":timing["token_id_match"],
                         "text_span":" ".join(x["token"] for x in observed),
                         "start_seconds":round(min(x["start_seconds"] for x in observed),4) if observed else "",
                         "end_seconds":round(max(x["end_seconds"] for x in observed),4) if observed else "",
                         "time_status":"aligned" if observed and timing["token_id_match"] else "unresolved"})
    save_csv(root/"attachment4_evidence_times_selected.csv",rows)
    for sample in summary:
        candidates=[r for r in rows if r["sample_id"]==sample["sample_id"] and
                    r["evidence_modality"]==sample["main_modality"] and r["time_status"]=="aligned"]
        if candidates:
            best=max(candidates,key=lambda r:r["target_logit_drop"])
            sample.update({"main_evidence_text":best["text_span"],
                           "main_evidence_start_s":best["start_seconds"],
                           "main_evidence_end_s":best["end_seconds"],
                           "main_evidence_logit_drop":best["target_logit_drop"]})
        else:
            sample.update({"main_evidence_text":"","main_evidence_start_s":"",
                           "main_evidence_end_s":"","main_evidence_logit_drop":""})
    save_csv(root/"attachment4_predictions_explanations_selected.csv",summary)
    save_json(root/"attachment4_alignment_audit_selected.json",{
        "samples":len(summary),"windows":len(rows),
        "timed_windows":sum(x["time_status"]=="aligned" for x in rows),
        "token_id_matches":sum(x["token_id_match"] for x in mapping.values())})
    print("evidence windows",len(rows),"timed",sum(x["time_status"]=="aligned" for x in rows))


if __name__=="__main__":
    main()
