from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd

class SurrogateModel:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        
    def train(self, df):
        """
        Trains the surrogate model on AGGREGATED weekly data for Binary Classification.
        df: DataFrame where each row is a week.
        Target: 'state_label' (0 or 1)
        """
        # Dynamic Feature Selection: Use all numeric columns except target and metadata
        exclude_cols = ['state_label', 'week_date', 'success_rate']
        available_features = [col for col in df.columns if col not in exclude_cols]
        
        # Verify we only have numeric features
        available_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
        
        print(f"Training on {len(available_features)} features: {available_features}")
        
        X = df[available_features].fillna(0)
        y = df['state_label']
        
        if len(X) < 10:
            print("Warning: Very few training samples (<10). Training on all data.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            preds = self.model.predict(X_test)
            print(f"Model trained. Accuracy: {accuracy_score(y_test, preds):.2f}")
            print(classification_report(y_test, preds))
            
        print("Feature Importances:")
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=available_features).sort_values(ascending=False)
            print(importances.head(10))
        
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def predict_state_class(self, platform_state):
        """
        Predicts binary state class (0 or 1) for a week.
        """
        if self.model is None:
            # Fallback heuristic
            A_t = platform_state.get('current_projects', 10)
            D_t = platform_state.get('current_projects_diversity', 1.0)
            
            # Simple heuristic: heavily loaded platform = bad (0), diverse = good (1)
            score = 0.5 - (A_t * 0.01) + (D_t * 0.1)
            return 1 if score > 0.5 else 0
            
        input_df = pd.DataFrame([platform_state])
        
        # Strict Validation
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = set(self.model.feature_names_in_)
            missing_cols = expected_cols - set(input_df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing features in input state: {missing_cols}")
                
            return self.model.predict(input_df[self.model.feature_names_in_])[0]
        else:
            return self.model.predict(input_df)[0]

    def predict_state_proba(self, platform_state):
        """
        Predicts probability of success (class 1).
        """
        if self.model is None:
            return 0.5
            
        input_df = pd.DataFrame([platform_state])
        
        if hasattr(self.model, 'feature_names_in_'):
            return self.model.predict_proba(input_df[self.model.feature_names_in_])[0][1]
        else:
             return self.model.predict_proba(input_df)[0][1]
