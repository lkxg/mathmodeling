"""Package the selected model, architecture comparisons, and final results."""
from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

from .core import OUT, ROOT


def main(target=None):
    target=target or ROOT/"outputs/E_q2_q3_improved.zip"
    experiment=OUT/"experiments"
    required=["selection.json","selected_metrics.json","selected_robustness.csv",
              "selected_error_analysis.json","attachment3_predictions_selected.csv",
              "attachment4_predictions_explanations_selected.csv",
              "attachment4_explanations_selected.json","attachment4_evidence_times_selected.csv",
              "attachment4_cards_selected.html","attachment4_input_audit_selected.json",
              "attachment4_alignment_audit_selected.json",
              "text_anchor.pt","shared_private.pt","reliability_proxy.pt"]
    for name in required:
        if not (experiment/name).is_file():raise FileNotFoundError(experiment/name)
    source=[p for p in (ROOT/"q23").rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and p.suffix in (".py",".md",".yaml",".txt")]
    source += [p for p in (ROOT/"q1").iterdir() if p.is_file() and
               p.suffix in (".py",".yaml",".txt",".sh")]
    include=[p for p in experiment.rglob("*") if p.is_file() and "experiment_cache" not in p.parts]
    include += [OUT/"normalization.npz",OUT/"attachment4_position_to_time.json",
                OUT/"model.pt",OUT/"metrics.json"]
    with ZipFile(target,"w",ZIP_DEFLATED,compresslevel=6) as z:
        for path in source+include:
            z.write(path,path.relative_to(ROOT))
    if target.stat().st_size>50_000_000:
        raise RuntimeError("Deliverable archive exceeds 50 MB")
    print(target,f"{target.stat().st_size/1024/1024:.2f} MiB")


if __name__=="__main__":
    main()
