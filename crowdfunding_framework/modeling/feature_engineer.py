import pandas as pd
import numpy as np
import datetime

class FeatureEngineer:
    def __init__(self, platform_launch_date='2010-01-01'):
        self.platform_launch_date = pd.to_datetime(platform_launch_date, utc=True)

    def calculate_entropy(self, categories):
        """
        Calculates Shannon entropy for a list of categories.
        categories: list or series of category IDs.
        Uses numpy for speed (avoids pd.Series overhead in hot loop).
        """
        if len(categories) == 0:
            return 0.0

        # Use numpy unique + counts instead of pd.Series.value_counts
        _, counts = np.unique(categories, return_counts=True)
        probs = counts / counts.sum()
        return -float(np.sum(probs * np.log(probs)))

    def calculate_hhi(self, contributions_amounts):
        """
        Calculates Herfindahl-Hirschman Index (HHI) for contributions amounts.
        Note: The notebook uses NORMALIZED HHI based on COUNTS.
        This generic method can compute HHI on any series.
        """
        total = sum(contributions_amounts)
        if total == 0:
            return 0.0
        
        shares = np.array(contributions_amounts) / total
        hhi = sum(shares ** 2)
        return hhi
    
    def calculate_normalized_hhi_from_counts(self, counts):
        """
        Notebook Logic: Normalized HHI based on contribution counts (shares of volume).
        HHI_norm = (HHI - 1/N) / (1 - 1/N)
        """
        total = sum(counts)
        if total == 0:
            return 0.0
            
        shares = np.array(counts) / total
        hhi = sum(shares ** 2)
        
        N = len(counts)
        if N > 1:
            hhi_norm = (hhi - 1/N) / (1 - 1/N)
        else:
            hhi_norm = 1.0 # Monopoly
            
        return hhi_norm

    def get_seasonality_features(self, date):
        """
        Returns a dictionary of seasonality features matching training logic.
        """
        # Ensure date is datetime
        if not isinstance(date, datetime.datetime) and not isinstance(date, pd.Timestamp):
             date = pd.to_datetime(date)
             
        m = date.month
        d = date.day
        
        # Helper for date ranges (Notebook uses strict inequalities > and <)
        def in_range(start_m, start_d, end_m, end_d):
            # Check if date is strictly between start and end dates
            # Logic: (m==start_m and d > start_d) OR (m > start_m and m < end_m) OR (m==end_m and d < end_d)
            # Handling wrap-around for Noel (Dec -> Jan) is special
            
            if start_m > end_m: # Wrap around (Noel: 12-21 to 01-08)
                # Date > 12-21 OR Date < 01-08
                current_after_start = (m == start_m and d > start_d) or (m > start_m)
                current_before_end = (m == end_m and d < end_d) or (m < end_m)
                return 1 if (current_after_start or current_before_end) else 0
            else:
                # Normal range
                # Greater than start
                after_start = (m == start_m and d > start_d) or (m > start_m)
                # Less than end
                before_end = (m == end_m and d < end_d) or (m < end_m)
                
                return 1 if (after_start and before_end) else 0

        features = {
            'winter': 1 if m in [1, 2, 3] else 0,
            'automn': 1 if m in [10, 11, 12] else 0, 
            'spring': 1 if m in [4, 5, 6] else 0,
            'summer': 1 if m in [7, 8, 9] else 0,
            'start_month': 1 if d <= 6 else 0,
            'end_month': 1 if d >= 26 else 0,
            'back_to_school': 1 if (m == 9) and (d < 15) else 0,
            # Notebook: date > .../10/19 and date < .../11/4
            'toussaint': in_range(10, 19, 11, 4),
            # Notebook: date > .../12/21 and date < .../01/8
            'noel': in_range(12, 21, 1, 8),  
            # Notebook: date > .../02/22 and date < .../03/10
            'winter_hol': in_range(2, 22, 3, 10),
            # Notebook: date > .../04/19 and date < .../05/05
            'spring_hol': in_range(4, 19, 5, 5),
            'start_year': 1 if m == 1 else 0,
            'end_year': 1 if m == 12 else 0
        }
        return features

    def compute_weekly_state(self, week_date, active_projects, new_projects):
        """
        Computes the full state vector S_t for a specific week (Online/GA usage).
        """
        
        # Ensure T is aware (skip conversion if already Timestamp)
        t = week_date if isinstance(week_date, pd.Timestamp) else pd.to_datetime(week_date, utc=True)
        t_end = t + pd.Timedelta(weeks=1)

        # 1. Segmentation
        starting_projs = new_projects

        # Ending: Filter from active_projects if data exists
        if not active_projects.empty and 'end_date' in active_projects.columns:
            ending_mask = (active_projects['end_date'] >= t) & (active_projects['end_date'] < t_end)
            ending_projs = active_projects[ending_mask]
        else:
            ending_projs = pd.DataFrame()
            
        # Current (Notebook definition: Union of start, end, intermediate)
        # Here we just use active_projects as the union superset
        current_projs = active_projects 
        
        # 2. Metrics Calculation Helper
        def compute_metrics(projs, prefix):
            count = len(projs)
            if count > 0:
                mean_goal = projs['goal'].mean()
                diversity = self.calculate_entropy(projs['category'])
            else:
                mean_goal = 0
                diversity = 0
            return {
                f'{prefix}': count,
                f'{prefix}_target': mean_goal,
                f'{prefix}_diversity': diversity
            }

        state = {}
        state.update(compute_metrics(current_projs, 'current_projects'))
        state.update(compute_metrics(starting_projs, 'starting_projects'))
        state.update(compute_metrics(ending_projs, 'ending_projects'))
        
        # 3. Contributions & Concentration
        # Removed Heuristics: total_contributions, concentration

        # 4. Lagged Success (Ending Success Rate)
        # Removed Heuristic: ending_success_rate (depended on estimated pledges)
            
        # 5. Seasonality
        seasonality = self.get_seasonality_features(t)
        state.update(seasonality)
        
        # 6. Platform Age (M_t)
        state['Age'] = int(t.year - 2010)
        
        return state

    def compute_history(self, projects_df, contributions_agg_df, start_date='2014-01-01', end_date='2024-01-01'):
        """
        Generates the weekly feature dataframe for training (Offline/Historical usage).
        projects_df: DataFrame of all projects using our standardized columns.
        contributions_agg_df: DataFrame of weekly contributions (count, amount).
        """
        print("Computing historical features from raw data...")
        
        # Ensure dates
        projects_df['start_date'] = pd.to_datetime(projects_df['start_date'], utc=True)
        # Est. End Date = Start + Duration (Fallback)
        if 'original_end_date' in projects_df.columns:
            projects_df['end_date'] = projects_df['original_end_date']
        else:
            projects_df['end_date'] = projects_df['start_date'] + pd.to_timedelta(projects_df['duration'], unit='D')
        
        # Prevent Leakage: Strict Filter on Inputs
        cutoff_date = pd.to_datetime(end_date, utc=True)
        # Filter projects that start significantly after the cutoff (buffer for active ones)
        # Actually, we rely on the loop window, but we can drop projects that start after end_date
        projects_df = projects_df[projects_df['start_date'] <= cutoff_date]
        
        # Filter contributions after cutoff
        if 'week_start' in contributions_agg_df.columns:
             contributions_agg_df = contributions_agg_df[contributions_agg_df['week_start'] <= cutoff_date]

        dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN', tz='UTC')
        
        history = []
        
        for t_idx, t in enumerate(dates):
            # Define window: t is Start of Week.
            t_end = t + pd.Timedelta(weeks=1)
            
            # --- 1. SEGMENTATION (Notebook Logic) ---
            # Starting: Start Date in [t, t_end)
            starting_mask = (projects_df['start_date'] >= t) & (projects_df['start_date'] < t_end)
            starting_projs = projects_df[starting_mask]
            
            # Ending: End Date in [t, t_end)
            ending_mask = (projects_df['end_date'] >= t) & (projects_df['end_date'] < t_end)
            ending_projs = projects_df[ending_mask]
            
            # Active (Intermediate): Start <= t AND End > t_end
            intermediate_mask = (projects_df['start_date'] <= t) & (projects_df['end_date'] > t_end)
            intermediate_projs = projects_df[intermediate_mask]
            
            # Current (Union)
            current_projs = pd.concat([starting_projs, ending_projs, intermediate_projs]).drop_duplicates(subset=['id'])
            
            # --- 2. CONTRIBUTIONS & CONCENTRATION (Notebook Logic) ---
            # REMOVED Heuristics to match Online mode
                
                
            # --- 3. DIVERSITY & TARGETS ---
            def compute_metrics(projs, prefix):
                count = len(projs)
                if count > 0:
                    mean_goal = projs['goal'].mean()
                    diversity = self.calculate_entropy(projs['category'])
                else:
                    mean_goal = 0
                    diversity = 0
                return {
                    f'{prefix}': count,
                    f'{prefix}_target': mean_goal, 
                    f'{prefix}_diversity': diversity
                }

            metrics = {}
            metrics.update(compute_metrics(current_projs, 'current_projects'))
            metrics.update(compute_metrics(starting_projs, 'starting_projects'))
            metrics.update(compute_metrics(ending_projs, 'ending_projects'))
            
            # --- 4. SUCCESS RATES ---
            
            # Target Vector (Y): Success Rate of STARTING projects
            if len(starting_projs) > 0:
                success_rate = starting_projs['status'].apply(lambda s: 1 if s == 'successful' else 0).mean()
            else:
                success_rate = np.nan

            # Lagged Feature: Success Rate of ENDING projects
            # REMOVED to match Online mode
                
            # --- 5. SEASONALITY (French Holidays) ---
            holidays = self.get_seasonality_features(t)
            
            # --- 6. PLATFORM AGE (M_t) ---
            # Age in years (Notebook Logic: Integer years since 2010)
            M_t = int(t.year - 2010)

            # Assemble Row (Aligning keys to reference input_features.csv)
            row = {
                'week_date': t,
                # 'total_contributions': total_contributions, # REMOVED
                **metrics,
                # 'concentration': HHI_norm, # REMOVED
                # 'ending_success_rate': success_rate_end, # REMOVED
                'success_rate': success_rate,
                'Age': M_t,
                **holidays
            }
            history.append(row)
            
        return pd.DataFrame(history)
