"""
Search for an 8-week window where the GA optimization yields:
  - improvement >= TARGET_IMPROVEMENT (%)
  - bad->good transitions >= TARGET_BAD_TO_GOOD

For each candidate window:
  - upcoming = historical projects whose original start_date falls in [W, W+horizon)
  - context  = projects active at W (start_date < W and end_date > W)
  - Run a lightweight GA, compute metrics, append a row to a CSV.

Run via:
  python find_window.py --stride 4 --weeks 8 --population 40 --generations 80

Stdout streams one line per window so it can be tailed with Monitor.
"""

import argparse
import csv
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Quiet down autogluon / sklearn at import
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

from crowdfunding_framework.data_loader import DataLoader
from crowdfunding_framework.modeling.surrogate_model import SurrogateModel
from crowdfunding_framework.optimization.optimization_flow import CrowdfundingProblem
from crowdfunding_framework.optimization.solver import GeneticSolver


def _compute_original_individual(window_projects_df, window_start, weeks):
    out = []
    for _, row in window_projects_df.iterrows():
        delta_days = (row['start_date'] - window_start).days
        wi = max(1, min(math.floor(delta_days / 7) + 1, weeks))
        out.append(wi)
    return out


def _classify_weeks(rates, median):
    return ['none' if np.isnan(r) else ('good' if r > median else 'bad') for r in rates]


def _count_bad_to_good(orig_rates, opt_rates, median):
    o = _classify_weeks(orig_rates, median)
    n = _classify_weeks(opt_rates, median)
    return sum(1 for a, b in zip(o, n) if a == 'bad' and b == 'good')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weeks', type=int, default=8)
    p.add_argument('--stride', type=int, default=4, help='Stride between window starts (weeks)')
    p.add_argument('--population', type=int, default=40)
    p.add_argument('--generations', type=int, default=80)
    p.add_argument('--min-projects', type=int, default=5)
    p.add_argument('--max-projects', type=int, default=150)
    p.add_argument('--target-improvement', type=float, default=10.0,
                   help='Required improvement %% to count as a hit')
    p.add_argument('--target-bad-to-good', type=int, default=3,
                   help='Required # of bad->good weeks to count as a hit')
    p.add_argument('--features-csv', type=str, default='weekly_features_from_raw.csv')
    p.add_argument('--output', type=str, default='window_search_results.csv')
    p.add_argument('--start-date', type=str, default=None,
                   help='Earliest window_start (YYYY-MM-DD)')
    p.add_argument('--end-date', type=str, default=None,
                   help='Latest window_start (YYYY-MM-DD)')
    args = p.parse_args()

    # Dataset median (the bad/good threshold)
    if not os.path.exists(args.features_csv):
        print(f"ERROR: {args.features_csv} missing", flush=True)
        sys.exit(1)
    feat_df = pd.read_csv(args.features_csv)
    median = float(feat_df['success_rate'].dropna().median())
    print(f"INIT median={median:.4f}", flush=True)

    # Load model
    model = SurrogateModel()
    if not model.load():
        print("ERROR: model not found; train first", flush=True)
        sys.exit(1)
    print("INIT model loaded", flush=True)

    # Load full historical projects
    loader = DataLoader()
    projects = loader.load_projects()
    if projects.empty or 'start_date' not in projects.columns:
        print("ERROR: no projects loaded", flush=True)
        sys.exit(1)
    projects = projects.dropna(subset=['start_date', 'goal', 'duration', 'category']).copy()
    projects['end_date'] = projects['start_date'] + pd.to_timedelta(projects['duration'], unit='D')
    print(f"INIT {len(projects)} projects spanning "
          f"{projects['start_date'].min().date()} -> {projects['start_date'].max().date()}",
          flush=True)

    # Window range
    min_d = pd.to_datetime(args.start_date, utc=True) if args.start_date else projects['start_date'].min()
    max_d = pd.to_datetime(args.end_date,   utc=True) if args.end_date   else projects['start_date'].max()
    # Snap to Mondays to keep windows aligned
    min_d = min_d.normalize() - pd.Timedelta(days=int(min_d.weekday()))
    max_d = max_d.normalize() - pd.Timedelta(days=int(max_d.weekday()))
    last_start = max_d - pd.Timedelta(weeks=args.weeks)
    starts = []
    cur = min_d
    while cur <= last_start:
        starts.append(cur)
        cur = cur + pd.Timedelta(weeks=args.stride)
    print(f"INIT scanning {len(starts)} windows (stride={args.stride}w, horizon={args.weeks}w)",
          flush=True)

    # Output CSV
    fieldnames = [
        'window_start', 'window_end', 'n_upcoming', 'n_context',
        'original_fitness', 'optimized_fitness', 'improvement_pct',
        'n_good_orig', 'n_good_opt', 'n_bad_to_good',
        'orig_mean_rate', 'opt_mean_rate', 'elapsed_s', 'is_hit',
    ]
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    hits = 0
    for i, ws in enumerate(starts, 1):
        we = ws + pd.Timedelta(weeks=args.weeks)
        t0 = time.time()

        upcoming = projects[(projects['start_date'] >= ws) & (projects['start_date'] < we)].copy()
        context  = projects[(projects['start_date'] < ws) & (projects['end_date'] > ws)].copy()

        n_up, n_ctx = len(upcoming), len(context)
        if n_up < args.min_projects or n_up > args.max_projects:
            print(f"[{i}/{len(starts)}] {ws.date()} SKIP n_up={n_up}", flush=True)
            continue

        try:
            upcoming = upcoming.reset_index(drop=True)
            context = context.reset_index(drop=True)

            orig_ind = _compute_original_individual(upcoming, ws, args.weeks)

            problem = CrowdfundingProblem(
                upcoming, model,
                start_date=ws,
                active_projects=context,
                time_horizon=args.weeks,
                deviation_weight=0.0,
                original_individual=orig_ind,
            )
            solver = GeneticSolver(
                problem,
                population_size=args.population,
                generations=args.generations,
            )
            solver.set_initial_individual(orig_ind)
            best, opt_fit, _ = solver.run()

            orig_fit = problem.evaluate(orig_ind)
            orig_det = problem.get_weekly_details(orig_ind)
            opt_det  = problem.get_weekly_details(best)

            orig_rates = np.array([d['predicted_rate'] for d in orig_det])
            opt_rates  = np.array([d['predicted_rate'] for d in opt_det])

            orig_class = _classify_weeks(orig_rates, median)
            opt_class  = _classify_weeks(opt_rates,  median)
            n_good_orig = sum(1 for c in orig_class if c == 'good')
            n_good_opt  = sum(1 for c in opt_class  if c == 'good')
            n_bg = _count_bad_to_good(orig_rates, opt_rates, median)

            imp = (opt_fit - orig_fit) / abs(orig_fit) * 100.0 if orig_fit else 0.0
            is_hit = (imp >= args.target_improvement) and (n_bg >= args.target_bad_to_good)
            if is_hit:
                hits += 1

            elapsed = time.time() - t0
            row = {
                'window_start': ws.date().isoformat(),
                'window_end':   we.date().isoformat(),
                'n_upcoming':   n_up,
                'n_context':    n_ctx,
                'original_fitness':  round(float(orig_fit), 5),
                'optimized_fitness': round(float(opt_fit), 5),
                'improvement_pct':   round(float(imp), 2),
                'n_good_orig':       n_good_orig,
                'n_good_opt':        n_good_opt,
                'n_bad_to_good':     n_bg,
                'orig_mean_rate':    round(float(np.nanmean(orig_rates)) if np.any(~np.isnan(orig_rates)) else 0.0, 4),
                'opt_mean_rate':     round(float(np.nanmean(opt_rates))  if np.any(~np.isnan(opt_rates))  else 0.0, 4),
                'elapsed_s':         round(elapsed, 1),
                'is_hit':            int(is_hit),
            }
            with open(args.output, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            tag = 'HIT' if is_hit else 'ok'
            print(f"[{i}/{len(starts)}] {ws.date()} n_up={n_up} n_ctx={n_ctx} "
                  f"orig={orig_fit:.4f} opt={opt_fit:.4f} imp={imp:+.1f}% "
                  f"good {n_good_orig}->{n_good_opt} b2g={n_bg} {elapsed:.1f}s {tag}",
                  flush=True)
        except Exception as e:
            print(f"[{i}/{len(starts)}] {ws.date()} ERROR {type(e).__name__}: {e}",
                  flush=True)

    print(f"DONE total_hits={hits} csv={args.output}", flush=True)


if __name__ == '__main__':
    main()
