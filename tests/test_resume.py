import csv
import json
from pathlib import Path

from scripts import run_pipeline
from rules import ga


def test_pipeline_resume_appends(tmp_path):
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    args_first = run_pipeline.build_parser().parse_args(
        [
            "--n",
            "2",
            "--k",
            "2",
            "--rule-bits",
            "111",
            "--no-spectral",
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(cache_dir),
            "--run-tag",
            "unit",
        ]
    )
    run_pipeline.run_pipeline(args_first)

    args_resume = run_pipeline.build_parser().parse_args(
        [
            "--n",
            "2",
            "--k",
            "2",
            "--rule-bits",
            "111",
            "--rule-bits",
            "101",
            "--no-spectral",
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(cache_dir),
            "--run-tag",
            "unit",
        ]
    )
    results = run_pipeline.run_pipeline(args_resume)

    jsonl_path = out_dir / "unit" / "summary.jsonl"
    ckpt_path = out_dir / "unit" / "checkpoint.json"
    assert jsonl_path.exists()
    assert ckpt_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert payload["payload"].get("completed") == 2

    bits_raw = {r["rule_bits_raw"] for r in results}
    assert bits_raw == {"111", "101"}


def test_ga_resume_adds_generations(tmp_path):
    out_csv = tmp_path / "ga_csv"
    cache_dir = tmp_path / "ga_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    conf_first = ga.GAConfig(
        pop_size=6,
        generations=1,
        device="cpu",
        use_lanczos=False,
        enable_spectral=False,
        progress_every=0,
        cache_dir=cache_dir,
    )
    _, front_path, gen_path = ga.ga_search_with_batch(2, 2, conf_first, out_csv_dir=str(out_csv), run_tag="resume")
    assert Path(front_path).exists()
    assert Path(gen_path).exists()
    # header + generation 0
    assert len(Path(gen_path).read_text(encoding="utf-8").splitlines()) == 2

    conf_resume = ga.GAConfig(
        pop_size=6,
        generations=2,
        device="cpu",
        use_lanczos=False,
        enable_spectral=False,
        progress_every=0,
        cache_dir=cache_dir,
    )
    _, front_resume, gen_resume = ga.ga_search_with_batch(2, 2, conf_resume, out_csv_dir=str(out_csv), run_tag="resume")

    gen_rows = list(csv.DictReader(Path(gen_resume).read_text(encoding="utf-8").splitlines()))
    assert {int(r["generation"]) for r in gen_rows} == {0, 1}
    front_rows = list(csv.DictReader(Path(front_resume).read_text(encoding="utf-8").splitlines()))
    assert {int(r["generation"]) for r in front_rows} == {0, 1}

    ckpt = json.loads((out_csv / "resume" / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["payload"].get("generation") == 1
    assert ckpt["payload"].get("target_generations") == 2
