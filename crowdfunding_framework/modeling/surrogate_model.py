from autogluon.tabular import TabularPredictor
from autogluon.common import space
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class SurrogateModel:
    def __init__(self, model_path='model_autogluon'):
        self.model = None
        self._cached_estimators = None
        self._cached_feature_names = None
        self._cached_rf = None          # Merged RF for fast batch prediction
        self._cached_expected_cols = None  # Column order for fast input prep

        # Resolve absolute path to project root to avoid CWD mismatches
        if model_path == 'model_autogluon':
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            self.model_path = os.path.join(base_dir, 'model_autogluon')
        else:
            self.model_path = model_path

    # ──────────────────────────────────────────────
    # STEP 1 — Feature Engineering
    # ──────────────────────────────────────────────
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches the dataset with lag, rolling, and interaction features
        before training. More signal = better R2.
        """
        df = df.copy().sort_values('week_date').reset_index(drop=True)

        # --- Lag features (what happened last week / 2 weeks ago) ---
        for lag in [1, 2, 3]:
            df[f'success_rate_lag{lag}'] = df['success_rate'].shift(lag)

        # --- Rolling statistics (trend over last N weeks) ---
        for window in [3, 5]:
            df[f'success_rate_roll_mean_{window}w'] = (
                df['success_rate'].shift(1).rolling(window).mean()
            )
            df[f'success_rate_roll_std_{window}w'] = (
                df['success_rate'].shift(1).rolling(window).std()
            )

        # --- Momentum: rate of change from last week ---
        df['success_rate_momentum'] = df['success_rate'].shift(1).diff()

        # --- Interaction features ---
        df['projects_diversity_interaction'] = (
            df['current_projects'] * df['current_projects_diversity']
        )
        df['starting_ending_ratio'] = (
            df['starting_projects'] / (df['ending_projects'] + 1e-6)
        )
        df['net_project_flow'] = df['starting_projects'] - df['ending_projects']

        # --- Drop rows with NaNs introduced by lags/rolling ---
        df = df.dropna().reset_index(drop=True)

        print(f"After feature engineering: {len(df)} rows, {len(df.columns)} columns")
        return df

    # ──────────────────────────────────────────────
    # STEP 2 — Hyperparameter Search Space
    # ──────────────────────────────────────────────
    def _get_rf_hyperparams(self):
        """
        Defines a rich HPO search space for RF.
        AutoGluon will sample from these distributions during tuning.
        """
        return {
            'RF': {
                # Number of trees: more = better but slower
                'n_estimators': space.Int(100, 1000, default=300),

                # Feature sampling per split
                'max_features': space.Categorical('sqrt', 'log2', 0.5, 0.7, 0.9),

                # Tree depth control — prevents overfitting on small data
                'max_depth': space.Categorical(None, 10, 20, 30),

                # Minimum samples required to split a node
                'min_samples_split': space.Int(2, 20, default=2),

                # Minimum samples at a leaf node
                'min_samples_leaf': space.Int(1, 10, default=1),

                # Bootstrap sampling
                'bootstrap': space.Categorical(True, False),
            }
        }

    # ──────────────────────────────────────────────
    # STEP 3 — Training
    # ──────────────────────────────────────────────
    def train(self, df: pd.DataFrame, exclude_features=None):
        """
        Full training pipeline:
          1. Clean up old model folder
          2. Feature engineering
          3. RF HPO with AutoGluon
          4. Bagging + stacking for better generalization
          5. Evaluation report

        exclude_features: list of feature names to drop before training
                          (e.g. ['Age'] to remove non-controllable features)
        """
        # Clean up old model folder to avoid "Learner is already fit" error
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
            print(f"Removed existing model at {self.model_path}")

        # Step 1: enrich features
        df = self._engineer_features(df)

        exclude_cols = {'week_date', 'success_rate'}
        if exclude_features:
            exclude_cols.update(exclude_features)
            print(f"Excluding features: {exclude_features}")

        available_features = [col for col in df.columns if col not in exclude_cols]
        available_features = [
            col for col in available_features
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        print(f"Training on {len(available_features)} features: {available_features}")

        train_df = df[available_features + ['success_rate']].fillna(0)

        if len(train_df) < 10:
            raise ValueError("Not enough training data")

        # Step 2: Fit with HPO enabled
        self.model = TabularPredictor(
            label='success_rate',
            path=self.model_path,
            eval_metric='r2',
            problem_type='regression',
        ).fit(
            train_data=train_df,
            hyperparameters=self._get_rf_hyperparams(),
            presets='best_v150',
            num_bag_folds=10,
            num_bag_sets=3,
            num_stack_levels=1,
            hyperparameter_tune_kwargs={
                'num_trials': 20,
                'scheduler': 'local',
                'searcher': 'auto',       # 'bayesopt' not supported with bagged models
            },
            time_limit=1800,
            verbosity=2,
        )

        # Step 3: Evaluation report
        self._evaluation_report(train_df)

    # ──────────────────────────────────────────────
    # STEP 4 — Evaluation Report
    # ──────────────────────────────────────────────
    def _evaluation_report(self, train_df: pd.DataFrame):
        leaderboard = self.model.leaderboard(silent=True)

        print("\n" + "=" * 60)
        print("MODEL LEADERBOARD")
        print("=" * 60)
        print(leaderboard[['model', 'score_val', 'fit_time']].to_string())

        best_r2 = leaderboard['score_val'].max()
        print(f"\nBest R2 Score (val): {best_r2:.4f}")

        if best_r2 < 0.3:
            print("⚠️  R2 still low — consider adding more historical data or features.")
        elif best_r2 < 0.6:
            print("⚠️  R2 moderate — model is learning but signal may be weak.")
        else:
            print("✅  R2 is good — model has strong predictive power.")

        print("\n" + "=" * 60)
        print("TOP FEATURE IMPORTANCES")
        print("=" * 60)
        try:
            importances = self.model.feature_importance(train_df)
            print(importances.head(15).to_string())
        except Exception as e:
            print(f"Could not compute feature importances: {e}")

    # ──────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────
    def load(self):
        if os.path.exists(self.model_path):
            self.model = TabularPredictor.load(self.model_path)
            return True
        return False

    # ──────────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────────
    def _prepare_input(self, platform_state: dict) -> pd.DataFrame:
        """Converts state dict to DataFrame, filling engineered features with 0 if missing."""
        input_df = pd.DataFrame([platform_state]).fillna(0)
        if hasattr(self.model, 'feature_metadata'):
            expected = self.model.feature_metadata.get_features()
            input_df = input_df.reindex(columns=expected, fill_value=0)
        return input_df

    def predict_success_rate(self, platform_state: dict) -> float:
        """Returns the mean prediction using the fast merged RF (bypasses AutoGluon overhead)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        merged_rf, feature_names = self._build_fast_rf()

        if self._cached_expected_cols is None:
            if feature_names is not None:
                self._cached_expected_cols = list(feature_names)
            else:
                self._cached_expected_cols = list(self.model.feature_metadata.get_features())

        X = np.array([[platform_state.get(col, 0.0) for col in self._cached_expected_cols]], dtype=np.float32)
        return float(merged_rf.predict(X)[0])

    def _get_rf_estimators(self):
        """
        Extracts all individual decision trees from all bagged RF folds.
        Results are cached after the first call to avoid repeated disk I/O.
        """
        # Return cached result if available
        if hasattr(self, '_cached_estimators') and self._cached_estimators is not None:
            return self._cached_estimators, self._cached_feature_names

        all_estimators = []
        feature_names = None

        for model_name in self.model.model_names():
            if 'RandomForest' not in model_name and 'RF' not in model_name:
                continue
            ag_model = self.model._trainer.load_model(model_name)

            if hasattr(ag_model, 'load_child') and hasattr(ag_model, 'models') and ag_model.models:
                for child_name in ag_model.models:
                    fold_model = ag_model.load_child(child_name)
                    sklearn_rf = fold_model.model
                    all_estimators.extend(sklearn_rf.estimators_)
                    if feature_names is None and hasattr(sklearn_rf, 'feature_names_in_'):
                        feature_names = sklearn_rf.feature_names_in_
            elif hasattr(ag_model, 'model'):
                sklearn_rf = ag_model.model
                all_estimators.extend(sklearn_rf.estimators_)
                if feature_names is None and hasattr(sklearn_rf, 'feature_names_in_'):
                    feature_names = sklearn_rf.feature_names_in_

        # Cache for subsequent calls
        self._cached_estimators = all_estimators
        self._cached_feature_names = feature_names
        print(f"Cached {len(all_estimators)} RF tree estimators for optimization.")

        return all_estimators, feature_names

    def _build_fast_rf(self):
        """
        Builds a single merged RandomForestRegressor containing ALL trees
        from all bagged folds. This lets us use sklearn's optimized C-level
        prediction instead of a Python loop over individual trees.
        """
        if self._cached_rf is not None:
            return self._cached_rf, self._cached_feature_names

        estimators, feature_names = self._get_rf_estimators()
        if not estimators:
            raise ValueError("No RF estimators found in the trained model.")

        # Create a shell RF and inject all trees
        merged_rf = RandomForestRegressor(n_estimators=len(estimators), warm_start=False)
        # Minimal fit to initialize internal structure (single dummy row)
        n_features = len(feature_names) if feature_names is not None else estimators[0].n_features_in_
        merged_rf.n_features_in_ = n_features
        merged_rf.estimators_ = estimators
        merged_rf.n_outputs_ = 1

        self._cached_rf = merged_rf
        print(f"Built fast merged RF with {len(estimators)} trees.")
        return merged_rf, feature_names

    def predict_success_rate_batch(self, states: list) -> np.ndarray:
        """Predicts success rates for multiple states at once (single RF call)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")
        if not states:
            return np.array([])

        merged_rf, feature_names = self._build_fast_rf()

        if self._cached_expected_cols is None:
            if feature_names is not None:
                self._cached_expected_cols = list(feature_names)
            else:
                self._cached_expected_cols = list(self.model.feature_metadata.get_features())

        cols = self._cached_expected_cols
        X = np.array([[s.get(col, 0.0) for col in cols] for s in states], dtype=np.float32)
        return merged_rf.predict(X)

    def predict_success_distribution(self, platform_state: dict):
        """
        Returns (mean, std) computed from ALL individual decision trees
        across all bagged RF folds — true tree-level uncertainty.

        Optimized: uses a merged RF for C-level vectorized prediction.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        merged_rf, feature_names = self._build_fast_rf()

        # Fast input preparation — avoid repeated DataFrame construction
        if self._cached_expected_cols is None:
            if feature_names is not None:
                self._cached_expected_cols = list(feature_names)
            else:
                self._cached_expected_cols = list(self.model.feature_metadata.get_features())

        # Build numpy array directly (skip DataFrame overhead)
        X = np.array([[platform_state.get(col, 0.0) for col in self._cached_expected_cols]])

        # Use tree_.predict (Cython) directly — avoids sklearn's Python wrapper overhead
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        tree_preds = np.empty(len(merged_rf.estimators_), dtype=np.float64)
        for i, tree in enumerate(merged_rf.estimators_):
            tree_preds[i] = tree.tree_.predict(X_c)[0, 0]

        return float(tree_preds.mean()), float(tree_preds.std())