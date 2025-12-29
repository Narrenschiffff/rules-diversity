import csv
import json
from pathlib import Path

from rules import motifs


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_analyze_local_features_extracts_labels_and_features(tmp_path: Path):
    xs = [1, 2, 3, 4]
    ys = [1.0, 2.5, 10.0, 5.0]
    bits_list = [
        "100011",  # idx 0
        "000101",  # idx 1 -> knee (robust_knee)
        "111111",  # idx 2 -> mur (mur_index)
        "010110",  # idx 3
    ]
    rows = []
    for i, (rc, y, bits) in enumerate(zip(xs, ys, bits_list)):
        rows.append(
            {
                "n": 3,
                "k": 3,
                "rule_bits": bits,
                "rule_count": rc,
                "Z_exact": y,
                "rows_m": rc + 1,
            }
        )

    front_csv = tmp_path / "front.csv"
    _write_rows(front_csv, rows)

    out_csv_dir = tmp_path / "out_csv"
    out_fig_dir = tmp_path / "figs"
    csv_out, json_out, figs = motifs.analyze_local_features(
        csv_paths=[str(front_csv)],
        out_csv_dir=str(out_csv_dir),
        out_fig_dir=str(out_fig_dir),
        style="default",
        logy=False,
    )

    csv_out = Path(csv_out)
    json_out = Path(json_out)
    assert csv_out.exists()
    assert json_out.exists()
    assert out_fig_dir.exists()

    records = json.loads(json_out.read_text(encoding="utf-8"))
    labels = {r["label"] for r in records}
    assert {"knee_current", "mur_current", "ymax_current", "ymin_current"} <= labels

    knee_rec = next(r for r in records if r["label"] == "knee_current")
    assert knee_rec["label_group"] == "knee"
    assert knee_rec["deg_sequence"] == "1;2;1"
    assert knee_rec["active_k"] == 3.0

    mur_rec = next(r for r in records if r["label"] == "mur_current")
    assert mur_rec["label_group"] == "mur"
    assert mur_rec["deg_histogram"]

    with open(csv_out, "r", encoding="utf-8") as f:
        rows_csv = list(csv.DictReader(f))
    assert len(rows_csv) == len(records)
    assert any(r["label"] == "ymin_current" for r in rows_csv)

    for fig in figs:
        assert Path(fig).exists()
