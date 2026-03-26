import pandas as pd
from crowdfunding_framework.modeling.surrogate_model import SurrogateModel
import math
import random
from crowdfunding_framework.modeling.feature_engineer import FeatureEngineer
from .solver import GeneticSolver
from .visualization import CrowdfundingVisualizer
import matplotlib.pyplot as plt
import numpy as np

class CrowdfundingProblem:
    """
    Encapsulates the specific constraints, data, and fitness evaluation
    for the Crowdfunding Scheduling problem.
    """
    def __init__(self, projects, surrogate_model, start_date=None, active_projects=None, time_horizon=8):
        self.projects = projects
        self.model = surrogate_model
        # Ensure status is lowercase for comparison if present
        if 'status' in self.projects.columns:
            self.projects['status'] = self.projects['status'].astype(str).str.lower()
            
        self.active_projects = active_projects if active_projects is not None else pd.DataFrame()
        self.T = time_horizon
        self.N = len(projects)
        self.fe = FeatureEngineer()
        
        # Base date
        if start_date is None:
            self.start_date = pd.to_datetime('2024-01-01', utc=True)
        else:
            self.start_date = pd.to_datetime(start_date, utc=True)
            
        # Bounds for the solver
        self.bounds = (1, self.T)
        
    def evaluate(self, individual):
        """
        Calculates fitness as the MEAN Predicted Weekly Success Rate over the horizon.
        """
        total_success_rate = 0
        valid_weeks = 0
        
        # 1. Pre-process schedule to find which projects are active in which week
        starts_by_week, active_by_week = self._precalculate_schedule_map(individual)

        # 2. Evaluate platform state for each week of the horizon
        for t in range(1, self.T + 1):
            current_date = self.start_date + pd.Timedelta(weeks=t-1)
            
            # Skip if no NEW projects are launching this week (Optimization Focus)
            new_indices = starts_by_week.get(t, [])
            if not new_indices:
                continue 
            
            # A. Prepare DataFrames
            active_projects_df = self._build_weekly_dataframe(t, current_date, individual, active_by_week)
            new_projects_df = self.projects.iloc[new_indices].copy()

            # B. Compute Feature Vector (S_t)
            state = self.fe.compute_weekly_state(current_date, active_projects_df, new_projects_df)
            
            # C. Predict Success (0 or 1)
            state_class = self.model.predict_state_class(state)
            
            total_success_rate += state_class
            valid_weeks += 1
            
        return total_success_rate if valid_weeks > 0 else 0.0

    def get_weekly_results(self, individual):
        """
        Returns the detailed list of predicted outcome (0 or 1) for each week.
        Indices 0 to T-1.
        """
        results = [0] * self.T
        
        # 1. Pre-process
        starts_by_week, active_by_week = self._precalculate_schedule_map(individual)
        
        # 2. Evaluate each week
        for t in range(1, self.T + 1):
             current_date = self.start_date + pd.Timedelta(weeks=t-1)
             
             new_indices = starts_by_week.get(t, [])
             if not new_indices:
                 results[t-1] = 0 # No launch = No success credit
                 continue
                 
             active_projects_df = self._build_weekly_dataframe(t, current_date, individual, active_by_week)
             new_projects_df = self.projects.iloc[new_indices].copy()
             
             state = self.fe.compute_weekly_state(current_date, active_projects_df, new_projects_df)
             state_class = self.model.predict_state_class(state)
             
             results[t-1] = state_class
             
        return results

    def get_weekly_probabilities(self, individual):
        """
        Returns the detailed list of predicted probabilities (class 1) for each week.
        Indices 0 to T-1.
        """
        results = [0.0] * self.T
        
        starts_by_week, active_by_week = self._precalculate_schedule_map(individual)
        
        for t in range(1, self.T + 1):
             current_date = self.start_date + pd.Timedelta(weeks=t-1)
             
             new_indices = starts_by_week.get(t, [])
             if not new_indices:
                 results[t-1] = 0.0
                 continue
                 
             active_projects_df = self._build_weekly_dataframe(t, current_date, individual, active_by_week)
             new_projects_df = self.projects.iloc[new_indices].copy()
             
             state = self.fe.compute_weekly_state(current_date, active_projects_df, new_projects_df)
             prob = self.model.predict_state_proba(state)
             
             results[t-1] = prob
             
        return results

    def get_actual_weekly_success_rate(self, individual):
        """
        Calculates the ACTUAL success rate (0.0-1.0) for each week based on 
        the 'status' column of projects starting in that week.
        Used for validation/comparison curve.
        """
        results = [0.0] * self.T
        
        # We only care about STARTS for this metric
        starts_by_week, _ = self._precalculate_schedule_map(individual)
        
        for t in range(1, self.T + 1):
             indices = starts_by_week.get(t, [])
             if not indices:
                 # No projects started this week -> Return NaN to show gap in plot
                 results[t-1] = np.nan
                 continue
                 
             # Check status of these projects
             week_projects = self.projects.iloc[indices]
             if 'status' not in week_projects.columns:
                 results[t-1] = 0.0
                 continue
                 
             # successful = 1, failed = 0
             successes = week_projects['status'].apply(lambda s: 1 if s == 'successful' else 0).sum()
             rate = successes / len(week_projects)
             results[t-1] = rate
             
        return results

    def _precalculate_schedule_map(self, individual):
        """Maps week_index -> [project_indices] for starts and activity."""
        starts = {t: [] for t in range(1, self.T + 1)}
        active = {t: [] for t in range(1, self.T + 1)}
        
        for i, start_week in enumerate(individual):
            duration_weeks = int(math.ceil(self.projects.iloc[i]['duration'] / 7))
            
            if start_week <= self.T:
                starts[start_week].append(i)
                
            # Mark active weeks [start, start + duration)
            end_week = min(start_week + duration_weeks, self.T + 1)
            for t in range(start_week, end_week):
                active[t].append(i)
                
        return starts, active

    def _build_weekly_dataframe(self, t, current_date, individual, active_by_week):
        """Combines Context (Real) + Candidates (GA) for a specific week."""
        # 1. Real Context (Already running)
        context_subset = pd.DataFrame()
        if not self.active_projects.empty:
            week_end = current_date + pd.Timedelta(weeks=1)
            mask = (self.active_projects['start_date'] < week_end) & \
                   (self.active_projects['end_date'] > current_date)
            context_subset = self.active_projects[mask].copy()

        # 2. GA Candidates (Virtual)
        ga_subset = pd.DataFrame()
        active_indices = active_by_week.get(t, [])
        
        if active_indices:
            ga_subset = self.projects.iloc[active_indices].copy()
            # Inject simulated dates
            self._inject_simulated_dates(ga_subset, individual, active_indices)


        # Combine
        return pd.concat([ga_subset, context_subset], ignore_index=True)


    def _inject_simulated_dates(self, df, individual, indices):
        """Calculates start/end dates based on the GA's week choices."""
        start_weeks = [individual[i] for i in indices]
        df['start_date'] = [self.start_date + pd.Timedelta(weeks=s-1) for s in start_weeks]
        
        # End date = Start + Duration
        if 'duration' in df.columns:
            df['end_date'] = df['start_date'] + pd.to_timedelta(df['duration'], unit='D')
        else:
            df['end_date'] = df['start_date'] + pd.Timedelta(days=30)

class OptimizationFlow:
    def run(self, args):
        print(f"--- Optimization Mode (Horizon: {args.weeks} weeks) ---")
        
        # Load Model
        model = SurrogateModel()
        if not model.load():
            print("Error: Model not found. Please run 'python main.py train' first.")
            return

        projects_df = pd.DataFrame()
        context_df = pd.DataFrame()
        sim_start_date = None
        
        # 1. Load Upcoming Projects
        if args.projects:
            print(f"Loading real upcoming projects from: {args.projects}")
            try:
                projects_df = pd.read_csv(args.projects)

                if 'Lancé le' in projects_df.columns:
                     projects_df['start_date'] = pd.to_datetime(projects_df['Lancé le'], utc=True)
                elif 'original_start_date' in projects_df.columns:
                     projects_df['start_date'] = pd.to_datetime(projects_df['original_start_date'], utc=True)
                
                required_map = {'ID projet': 'id', 'Objectif de collecte': 'goal', 'Durée de la collecte': 'duration', 'Catégorie principale': 'category'}
                projects_df.rename(columns=required_map, inplace=True)
                
                if 'goal' in projects_df.columns: projects_df['goal'] = pd.to_numeric(projects_df['goal'], errors='coerce')
                if 'duration' in projects_df.columns: projects_df['duration'] = pd.to_numeric(projects_df['duration'], errors='coerce')
                
                projects_df = projects_df.dropna(subset=['start_date'])
                if not projects_df.empty:
                    sim_start_date = projects_df['start_date'].min()
                    print(f"Simulation Start Date inferred: {sim_start_date.date()}")
                
            except Exception as e:
                print(f"Error loading projects file: {e}")
                return
        
        # 2. Load Context (Active Projects)
        if args.context:
            print(f"Loading context (active projects) from: {args.context}")
            try:
                context_df = pd.read_csv(args.context)
                # Helper to standardize
                if 'Lancé le' in context_df.columns:
                     context_df['start_date'] = pd.to_datetime(context_df['Lancé le'], utc=True)
                elif 'original_start_date' in context_df.columns:
                     context_df['start_date'] = pd.to_datetime(context_df['original_start_date'], utc=True)
                
                if 'Fini le' in context_df.columns:
                     context_df['end_date'] = pd.to_datetime(context_df['Fini le'], utc=True)
                elif 'original_end_date' in context_df.columns:
                     context_df['end_date'] = pd.to_datetime(context_df['original_end_date'], utc=True)
                     
                required_map = {'ID projet': 'id', 'Objectif de collecte': 'goal', 'Durée de la collecte': 'duration', 'Catégorie principale': 'category'}
                context_df.rename(columns=required_map, inplace=True)
                
                if 'duration' in context_df.columns: context_df['duration'] = pd.to_numeric(context_df['duration'], errors='coerce')
                
                # If end_date missing, infer
                if 'end_date' not in context_df.columns and 'start_date' in context_df.columns and 'duration' in context_df.columns:
                    context_df['end_date'] = context_df['start_date'] + pd.to_timedelta(context_df['duration'], unit='D')
                    
                # Ensure we have start/end
                context_df = context_df.dropna(subset=['start_date', 'end_date'])
                print(f"Loaded {len(context_df)} active context projects.")
                
            except Exception as e:
                print(f"Error loading context file: {e}")
                return

        if projects_df.empty:
            print("Error: No projects loaded. You must provide upcoming projects via --projects.")
            return
        
        problem = CrowdfundingProblem(
            projects_df, 
            model,
            start_date=sim_start_date,
            active_projects=context_df,
            time_horizon=args.weeks
        )
        
        # Calculate Original Schedule (Initial Individual)
        original_ind = []
        for _, row in projects_df.iterrows():
            if pd.notna(row['start_date']) and sim_start_date:
                delta_days = (row['start_date'] - sim_start_date).days
                week_idx = math.floor(delta_days / 7) + 1
                # Clamp to bounds (though original might be outside, we clamp for valid evaluation)
                week_idx = max(1, min(week_idx, args.weeks))
                original_ind.append(week_idx)
            else:
                original_ind.append(1) # Fallback

        solver = GeneticSolver(
            problem,
            population_size=args.population,
            generations=args.generations
        )
        solver.set_initial_individual(original_ind)
        
        best, fitness, history = solver.run()
        print(f"Optimization Complete. Best Fitness (Good Weeks): {fitness}")
        
        # --- Visualization & Comparison ---
        
        # 1. Evaluate Original Schedule

        original_fitness = problem.evaluate(original_ind)
        print(f"Original Schedule Fitness: {original_fitness}")
        
        improvement = ((fitness - original_fitness) / original_fitness * 100) if original_fitness > 0 else 0
        print(f"Improvement: {improvement:.1f}%")

        # 2. Plotting
        plt.figure(figsize=(10, 6))
        
        # Subplot 1: Convergence
        plt.subplot(1, 2, 1)
        plt.plot(history, label='Best Fitness')
        plt.title('Genetic Algorithm Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Total Success)')
        plt.grid(True)
        plt.legend()
        
        # Subplot 2: Comparison (Total)
        plt.subplot(1, 2, 2)
        bars = plt.bar(['Original', 'Optimized'], [original_fitness, fitness], color=['gray', 'green'])
        plt.title('Total Schedule Comparison')
        plt.ylabel('Predicted Success Score')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
            
        # Always calculate Predicted Probability for Original Schedule (for fair comparison)
        orig_probs = problem.get_weekly_probabilities(original_ind)
        opt_probs = problem.get_weekly_probabilities(best)
        
        # Subplot 3: Week-by-Week Comparison (Curves) - REMOVED per user request
        
        # Calculate Actual Rate for Original (if available) for reference
        orig_actual = None
        if 'status' in projects_df.columns:
            orig_actual = problem.get_actual_weekly_success_rate(original_ind)

        plt.tight_layout()
        plt.savefig('optimization_results.png')
        plt.savefig('optimization_results.png')
        print("Results saved to 'optimization_results.png'")

        # 3. Interactive Visualization (Plotly)
        try:
            print("Generating Interactive Report...")
            viz = CrowdfundingVisualizer(projects_df, args.weeks)
            
            # A. Gantt
            fig_gantt = viz.compare_schedules_gantt(original_ind, best, sim_start_date)
            
            # B. Success Probabilities
            # Pass (Optimized, Original Predicted, Original Actual)
            # User requested to REMOVE Original Actual (Gray curve)
            fig_probs = viz.plot_success_probabilities(opt_probs, orig_probs, None)
            
            # C. Heatmap (Example: Weekly Load/Success)
            # We construct a simple matrix: Row 1 = Orig Pred, Row 2 = Opt Pred
            heatmap_data = [orig_probs, opt_probs]
            fig_heatmap = viz.plot_weekly_heatmap(heatmap_data)
            
            viz.save_report([fig_gantt, fig_probs, fig_heatmap], filename="optimization_report.html")
            print("Interactive report saved to 'optimization_report.html'")
            
        except Exception as e:
            print(f"Error generating interactive report: {e}")

