# 规则–模式多目标搜索与膝点分析

> 在 \(n\times n\) 环面格上，用 \(k\) 种状态的局部相容规则 \(R\) 生成全局模式，
> 在“最大化模式规模/生长率（\(Z_n\) 或 \(\widehat{\mathrm{trace}}(T^n)\)）”与“最小化规则复杂度（\(|R|\)）”之间寻找帕累托前沿，并识别**膝点**。

- Stage-1：广域扫描（同时输出 **raw** 与 **canon** 口径的前沿）。
- GA：NSGA-II（带**父本对齐**、**canonical 折叠**与**评估缓存**）。
- Symmetry：在膝点代表上统计模式**几何/重标**去重。
- Viz：同图对比 `stage1_raw / stage1_canon / ga_canon`，自动标注二阶差分与折线距离两种膝点。

![Quicklook](docs/_assets/quicklook.png)



## 快速开始（Bash）
```bash
python -u scripts/rd_cli.py stage1 --n 4 --k 4 --out-csv notebooks/results/out_csv --verbosity 2

python -u scripts/rd_cli.py ga --n 4 --k 4 --device cuda   --pop-size 48 --generations 60 --p-mut 0.08 --p-cx 0.85 --elite-keep 8   --r-vals 2 --power-iters 40 --trace-mode hutchpp --hutch-s 24   --seed-from-stage1 --max-stage1-seeds 400 --progress-every 8 --verbosity 2

python -u scripts/rd_cli.py viz-nk --n 4 --k 4   --front notebooks/results/out_csv/stage1_pareto_n4_k4_raw.csv           notebooks/results/out_csv/stage1_pareto_n4_k4_canon.csv           notebooks/results/out_csv/pareto_front_n4_k4_ga.csv   --out-dir notebooks/results/figs --y-log --style ieee

python -u scripts/rd_cli.py motifs --examples notebooks/results/out_csv/motifs/motif_knee_examples.csv   --out-csv notebooks/results/out_csv/motifs --out-dir notebooks/results/figs --style ieee

python -u scripts/rd_cli.py symmetry --front notebooks/results/out_csv/stage1_pareto_n4_k4_canon.csv   --n 4 --k 4 --geo rot,ref,trans --state-perm --samples 6   --out-csv notebooks/results/symmetry --out-dir notebooks/results/figs --style ieee
```

## 快速开始（PowerShell）
```powershell
python -u scripts/rd_cli.py stage1 `
  --n 4 --k 4 `
  --out-csv notebooks\results\out_csv `
  --verbosity 2

python -u scripts/rd_cli.py ga `
  --n 4 --k 4 `
  --device cuda `
  --pop-size 48 --generations 60 `
  --p-mut 0.08 --p-cx 0.85 --elite-keep 8 `
  --r-vals 2 --power-iters 40 --trace-mode hutchpp --hutch-s 24 `
  --seed-from-stage1 --max-stage1-seeds 400 `
  --progress-every 8 --verbosity 2

python -u scripts/rd_cli.py viz-nk `
  --n 4 --k 4 `
  --front notebooks\results\out_csv\stage1_pareto_n4_k4_raw.csv `
         notebooks\results\out_csv\stage1_pareto_n4_k4_canon.csv `
         notebooks\results\out_csv\pareto_front_n4_k4_ga.csv `
  --out-dir notebooks\results\figs --y-log --style ieee

python -u scripts/rd_cli.py motifs `
  --examples notebooks\results\out_csv\motifs\motif_knee_examples.csv `
  --out-csv notebooks\results\out_csv\motifs `
  --out-dir notebooks\results\figs --style ieee

python -u scripts/rd_cli.py symmetry `
  --front notebooks\results\out_csv\stage1_pareto_n4_k4_canon.csv `
  --n 4 --k 4 --geo rot,ref,trans --state-perm --samples 6 `
  --out-csv notebooks\results\symmetry `
  --out-dir notebooks\results\figs --style ieee
```

---

## 目录结构（要点）
```
rules_diversity_final/
├─ rules/                # stage1, ga, viz, symmetry, motifs, spectrum, eval, utils...
├─ scripts/rd_cli.py     # 统一命令行入口
├─ notebooks/results/    # out_csv / figs / symmetry / motifs
└─ README.md
```
