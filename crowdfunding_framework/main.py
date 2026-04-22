import argparse
import sys
import os

# Ensure project root is in path for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    parser = argparse.ArgumentParser(description="Crowdfunding Optimization Framework")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- TRAIN Command ---
    train_parser = subparsers.add_parser('train', help='Train the model (computes features if needed)')
    train_parser.add_argument('--force', action='store_true', help='Force re-computation of features')
    train_parser.add_argument('--features', type=str, default='weekly_features_from_raw.csv', help='Path to feature file')
    train_parser.add_argument('--start-date', type=str, help='Start date for training data (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', type=str, help='End date for training data (YYYY-MM-DD)')
    train_parser.add_argument('--exclude-features', type=str, nargs='+', default=None, dest='exclude_features', help='Features to exclude from training (e.g. Age winter summer)')

    # --- FEATURES Command ---
    feat_parser = subparsers.add_parser('features', help='Run feature engineering only')
    feat_parser.add_argument('--output', type=str, default='weekly_features_from_raw.csv', help='Output path for features')
    
    # --- OPTIMIZE Command ---
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--population', type=int, default=10, help='GA Population Size')
    opt_parser.add_argument('--generations', type=int, default=30, help='GA Generations')
    opt_parser.add_argument('--projects', type=str, help='Path to upcoming projects CSV (Real IO)')
    opt_parser.add_argument('--context', type=str, help='Path to currently active projects CSV (Context)')
    opt_parser.add_argument('--weeks', type=int, default=8, help='Optimization Horizon (Weeks)')
    opt_parser.add_argument('--deviation-weight', type=float, default=0.01, dest='deviation_weight', help='Weight for penalizing L1 deviation from the original schedule')

    # --- PARETO Command ---
    pareto_parser = subparsers.add_parser('pareto', help='Sweep deviation weights and plot Pareto front')
    pareto_parser.add_argument('--population', type=int, default=10, help='GA Population Size')
    pareto_parser.add_argument('--generations', type=int, default=30, help='GA Generations')
    pareto_parser.add_argument('--projects', type=str, help='Path to upcoming projects CSV')
    pareto_parser.add_argument('--context', type=str, help='Path to currently active projects CSV')
    pareto_parser.add_argument('--weeks', type=int, default=8, help='Optimization Horizon (Weeks)')
    pareto_parser.add_argument('--weights', type=float, nargs='+', default=[0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0], dest='pareto_weights', help='Deviation weights to sweep')

    # --- ANALYZE Command ---
    analyze_parser = subparsers.add_parser('analyze', help='Analyze historical data to find best periods to optimize')
    analyze_parser.add_argument('--features', type=str, default='weekly_features_from_raw.csv', help='Path to feature file')
    analyze_parser.add_argument('--weeks', type=int, default=8, help='Optimization horizon to evaluate')
    analyze_parser.add_argument('--top', type=int, default=10, help='Number of top periods to show')

    # --- EXTRACT Command ---
    ext_parser = subparsers.add_parser('extract', help='Extract Active/Upcoming projects from raw data')
    ext_parser.add_argument('--date', type=str, required=True, help='Reference Date (YYYY-MM-DD)')
    ext_parser.add_argument('--weeks', type=int, default=8, help='Horizon for upcoming projects')
    ext_parser.add_argument('--output', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    if args.command == 'train':
        print("--- Training Mode ---")
        from crowdfunding_framework.modeling.trainer import ModelTrainer
        trainer = ModelTrainer(feature_file=args.features)
        trainer.train(force_compute=args.force, start_date=args.start_date, end_date=args.end_date, exclude_features=args.exclude_features)

    elif args.command == 'features':
        print("--- Feature Engineering Mode ---")
        from crowdfunding_framework.modeling.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.compute_features(output_path=args.output)

    elif args.command == 'optimize':
        from crowdfunding_framework.optimization.optimization_flow import OptimizationFlow
        flow = OptimizationFlow()
        flow.run(args)

    elif args.command == 'analyze':
        from crowdfunding_framework.optimization.period_analyzer import PeriodAnalyzer
        analyzer = PeriodAnalyzer(feature_file=args.features)
        analyzer.run(horizon=args.weeks, top_n=args.top)

    elif args.command == 'pareto':
        from crowdfunding_framework.optimization.optimization_flow import OptimizationFlow
        flow = OptimizationFlow()
        flow.run_pareto(args)

    elif args.command == 'extract':
        from crowdfunding_framework.tools.extraction_flow import ExtractionFlow
        flow = ExtractionFlow()
        flow.run(args)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
