from crowdfunding_framework.data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .surrogate_model import SurrogateModel
import pandas as pd
import os

class ModelTrainer:
    def __init__(self, data_dir='data/raw data', feature_file='weekly_features_from_raw.csv', model_file='model_autogluon'):
        self.data_dir = data_dir
        self.feature_file = feature_file
        self.model_file = model_file
        self.loader = DataLoader(data_dir)
        self.fe = FeatureEngineer()
        self.model = SurrogateModel(model_path=model_file)
        
    def train(self, force_compute=False):
        """
        Orchestrates the training pipeline.
        1. Checks for cached features.
        2. If missing or forced, computes features from raw.
        3. Trains model.
        """
        train_df = None
        
        # 1. Check Cache
        if not force_compute and os.path.exists(self.feature_file):
            print(f"Index found: {self.feature_file}. Loading features directly...")
            train_df = pd.read_csv(self.feature_file)
            
            # Simple validation
            if 'success_rate' not in train_df.columns:
                print("Cached file invalid (missing target). Recomputing...")
                train_df = None
        
    def compute_features(self, output_path=None, start_date=None, end_date=None):
        """
        Computes features from raw data and saves to CSV.
        """
        target_file = output_path if output_path else self.feature_file
        
        print("Computing features from raw data...")
        print("1. Loading Projects...")
        projects_df = self.loader.load_projects()
        if projects_df.empty:
            print("Error: No projects found.")
            return pd.DataFrame()
            
        print("2. Loading Contributions...")
        conts_df = self.loader.load_contributions_aggregated()
        
        print("3. Engineering Features...")
        min_date = projects_df['start_date'].min()
        max_date = projects_df['start_date'].max()
        
        # Use provided dates or default to min/max
        use_start = start_date if start_date else str(min_date.date())
        use_end = end_date if end_date else str(max_date.date())
        
        history_df = self.fe.compute_history(
            projects_df, 
            conts_df, 
            start_date=use_start, 
            end_date=use_end
        )
        
        # Filter for valid rows if needed (e.g. have a target)
        # But for just feature engineering, maybe we keep all? 
        # The training logic did: `train_df = history_df.dropna(subset=['success_rate'])`
        # Let's keep it generic here, but maybe dropna for safety if 'success_rate' is key.
        final_df = history_df.dropna(subset=['success_rate'])
        
        print(f"Saving computed features to {target_file}")
        final_df.to_csv(target_file, index=False)
        return final_df

    def train(self, force_compute=False, start_date=None, end_date=None, exclude_features=None):
        """
        Orchestrates the training pipeline.
        1. Checks for cached features.
        2. If missing or forced, computes features from raw.
        3. Trains model.

        exclude_features: list of feature names to exclude from training
                          (e.g. ['Age'] to force the model to rely on controllable features)
        """
        train_df = None

        # 1. Check Cache
        if not force_compute and os.path.exists(self.feature_file):
            print(f"Index found: {self.feature_file}. Loading features directly...")
            train_df = pd.read_csv(self.feature_file)

            # Simple validation
            if 'success_rate' not in train_df.columns:
                print("Cached file invalid (missing target). Recomputing...")
                train_df = None

        # 2. Compute if needed
        if train_df is None:
            train_df = self.compute_features(start_date=start_date, end_date=end_date)

        # 3. Train
        if not train_df.empty:
            print("Training Surrogate Model...")
            self.model.train(train_df, exclude_features=exclude_features)
        else:
            print("Error: No valid training data available.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
