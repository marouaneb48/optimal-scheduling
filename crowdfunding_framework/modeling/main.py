import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crowdfunding_framework.modeling.trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Crowdfunding Modeling CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)

    # --- TRAIN Command ---
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--force', action='store_true', help='Force re-computation')
    train_parser.add_argument('--features', type=str, default='weekly_features_from_raw.csv', help='Path to feature file')

    # --- FEATURES Command ---
    feat_parser = subparsers.add_parser('features', help='Run feature engineering only')
    feat_parser.add_argument('--output', type=str, default='weekly_features_from_raw.csv', help='Output path')

    args = parser.parse_args()

    if args.command == 'train':
        print("--- Training Mode ---")
        trainer = ModelTrainer(feature_file=args.features)
        trainer.train(force_compute=args.force)

    elif args.command == 'features':
        print("--- Feature Engineering Mode ---")
        trainer = ModelTrainer()
        trainer.compute_features(output_path=args.output)

if __name__ == "__main__":
    main()
