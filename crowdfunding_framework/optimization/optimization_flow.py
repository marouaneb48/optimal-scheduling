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
    def __init__(self, projects, surrogate_model, start_date=None, active_projects=None, time_horizon=8, deviation_weight=0, original_individual=None):
        self.projects = projects
        self.model = surrogate_model
        self.deviation_weight = deviation_weight
        self.original_individual = np.array(original_individual) if original_individual is not None else None
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

        # Fitness cache: avoid re-evaluating identical schedules
        self._fitness_cache = {}

        # Pre-extract numpy arrays from projects (avoid pandas in hot loop)
        self._proj_durations_weeks = np.ceil(self.projects['duration'].values / 7).astype(int)
        self._proj_goals = self.projects['goal'].values.astype(float)
        self._proj_categories = self.projects['category'].values

        # Pre-compute week dates and seasonality (these never change)
        self._week_dates = [self.start_date + pd.Timedelta(weeks=t-1) for t in range(1, self.T + 1)]
        self._week_seasonality = [self.fe.get_seasonality_features(d) for d in self._week_dates]
        self._week_ages = [int(d.year - 2010) for d in self._week_dates]

        # Pre-compute context metrics per week (counts, mean_goal, diversity)
        self._context_cache = {}
        self._context_metrics = {}
        if not self.active_projects.empty:
            ctx_goals = self.active_projects['goal'].values
            ctx_cats = self.active_projects['category'].values
            ctx_starts = self.active_projects['start_date'].values
            ctx_ends = self.active_projects['end_date'].values

            for t in range(1, self.T + 1):
                current_date = self._week_dates[t-1]
                week_end = current_date + pd.Timedelta(weeks=1)
                cd_np = np.datetime64(current_date)
                we_np = np.datetime64(week_end)

                mask = (ctx_starts < we_np) & (ctx_ends > cd_np)
                count = int(mask.sum())

                # Ending context projects this week
                ending_mask = (ctx_ends >= cd_np) & (ctx_ends < we_np)
                ending_count = int(ending_mask.sum())
                if ending_count > 0:
                    ending_goals = ctx_goals[ending_mask]
                    ending_cats = ctx_cats[ending_mask]
                else:
                    ending_goals = np.array([])
                    ending_cats = np.array([])

                if count > 0:
                    active_goals = ctx_goals[mask]
                    active_cats = ctx_cats[mask]
                    self._context_metrics[t] = {
                        'count': count,
                        'mean_goal': float(active_goals.mean()),
                        'diversity': self.fe.calculate_entropy(active_cats),
                        'goals': active_goals,
                        'categories': active_cats,
                        'ending_count': ending_count,
                        'ending_goals': ending_goals,
                        'ending_cats': ending_cats,
                    }
                else:
                    self._context_metrics[t] = {
                        'count': 0, 'mean_goal': 0, 'diversity': 0,
                        'goals': np.array([]), 'categories': np.array([]),
                        'ending_count': ending_count,
                        'ending_goals': ending_goals,
                        'ending_cats': ending_cats,
                    }
                # Populate the DataFrame cache used by the reporting/plotting path
                self._context_cache[t] = self.active_projects[mask].copy() if count > 0 else pd.DataFrame()
            print(f"Pre-cached context metrics for {self.T} weeks.")
        
    def evaluate(self, individual):
        """
        Fitness = mean_predicted_success_rate - deviation_weight * mean_L1_deviation
        where L1 deviation = (1/N) * Σ |individual[i] - original[i]|
        """
        cache_key = tuple(individual)
        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]

        # Fast schedule map using pre-extracted numpy arrays
        starts = {t: [] for t in range(1, self.T + 1)}
        active = {t: [] for t in range(1, self.T + 1)}
        endings = {t: [] for t in range(1, self.T + 1)}
        for i, start_week in enumerate(individual):
            if start_week <= self.T:
                starts[start_week].append(i)
            end_week = min(start_week + self._proj_durations_weeks[i], self.T + 1)
            for t in range(start_week, end_week):
                active[t].append(i)
            # Last active week is end_week-1; project "ends" there (mirrors context logic)
            ga_end_week = end_week - 1
            if 1 <= ga_end_week <= self.T:
                endings[ga_end_week].append(i)

        # Collect all weekly states as dicts, then batch-predict
        states = []
        for t in range(1, self.T + 1):
            new_idx = starts[t]
            if not new_idx:
                continue
            state = self._compute_state_fast(t, new_idx, active[t], endings[t])
            states.append(state)

        if states:
            predictions = self.model.predict_success_rate_batch(states)
            mean_rate = float(predictions.sum()) / max(1, len(states))
        else:
            mean_rate = 0.0

        # L1 deviation penalty from original schedule
        deviation = 0.0
        if self.original_individual is not None:
            deviation = np.abs(np.array(individual) - self.original_individual).mean()

        result = mean_rate - self.deviation_weight * deviation
        self._fitness_cache[cache_key] = result
        return result

    def _compute_state_fast(self, t, new_indices, active_indices, ending_indices=None):
        """
        Computes the feature state dict for week t without any pandas operations.
        Uses pre-extracted numpy arrays and cached context metrics.
        """
        # --- GA projects: active, starting, ending ---
        ga_goals_active = self._proj_goals[active_indices]
        ga_cats_active = self._proj_categories[active_indices]

        ga_goals_new = self._proj_goals[new_indices]
        ga_cats_new = self._proj_categories[new_indices]

        # Context (pre-computed)
        ctx = self._context_metrics.get(t)
        if ctx and ctx['count'] > 0:
            all_active_goals = np.concatenate([ga_goals_active, ctx['goals']])
            all_active_cats = np.concatenate([ga_cats_active, ctx['categories']])
        else:
            all_active_goals = ga_goals_active
            all_active_cats = ga_cats_active

        # Ending projects: GA endings + context endings
        ga_end_idx = ending_indices or []
        ga_ending_goals = self._proj_goals[ga_end_idx] if ga_end_idx else np.array([])
        ga_ending_cats = self._proj_categories[ga_end_idx] if ga_end_idx else np.array([])

        if ctx and ctx['ending_count'] > 0:
            all_ending_goals = np.concatenate([ga_ending_goals, ctx['ending_goals']])
            all_ending_cats = np.concatenate([ga_ending_cats, ctx['ending_cats']])
        else:
            all_ending_goals = ga_ending_goals
            all_ending_cats = ga_ending_cats

        ending_count = len(all_ending_goals)
        ending_mean_goal = float(all_ending_goals.mean()) if ending_count > 0 else 0
        ending_diversity = self.fe.calculate_entropy(all_ending_cats) if ending_count > 0 else 0

        current_count = len(all_active_goals)
        current_mean_goal = float(all_active_goals.mean()) if current_count > 0 else 0
        current_diversity = self.fe.calculate_entropy(all_active_cats) if current_count > 0 else 0

        starting_count = len(new_indices)
        starting_mean_goal = float(ga_goals_new.mean())
        starting_diversity = self.fe.calculate_entropy(ga_cats_new)

        state = {
            'current_projects': current_count,
            'current_projects_target': current_mean_goal,
            'current_projects_diversity': current_diversity,
            'starting_projects': starting_count,
            'starting_projects_target': starting_mean_goal,
            'starting_projects_diversity': starting_diversity,
            'ending_projects': ending_count,
            'ending_projects_target': ending_mean_goal,
            'ending_projects_diversity': ending_diversity,
            'Age': self._week_ages[t - 1],
        }
        state.update(self._week_seasonality[t - 1])
        return state

    def decompose_fitness(self, individual):
        """
        Returns (mean_success_rate, mean_L1_deviation) for a given schedule.
        Useful for plotting the Pareto front.
        """
        starts = {t: [] for t in range(1, self.T + 1)}
        active = {t: [] for t in range(1, self.T + 1)}
        endings = {t: [] for t in range(1, self.T + 1)}
        for i, start_week in enumerate(individual):
            if start_week <= self.T:
                starts[start_week].append(i)
            end_week = min(start_week + self._proj_durations_weeks[i], self.T + 1)
            for t in range(start_week, end_week):
                active[t].append(i)
            ga_end_week = end_week - 1
            if 1 <= ga_end_week <= self.T:
                endings[ga_end_week].append(i)

        states = []
        for t in range(1, self.T + 1):
            new_idx = starts[t]
            if not new_idx:
                continue
            states.append(self._compute_state_fast(t, new_idx, active[t], endings[t]))

        if states:
            predictions = self.model.predict_success_rate_batch(states)
            mean_rate = float(predictions.sum()) / max(1, len(states))
        else:
            mean_rate = 0.0

        deviation = 0.0
        if self.original_individual is not None:
            deviation = float(np.abs(np.array(individual) - self.original_individual).mean())

        return mean_rate, deviation

    def get_weekly_results(self, individual):
        """
        Returns the detailed list of predicted outcome (success rate) for each week.
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
             state_rate = self.model.predict_success_rate(state)
             
             results[t-1] = state_rate
             
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
             prob = self.model.predict_success_rate(state)
             
             results[t-1] = prob
             
        return results

    def get_weekly_details(self, individual):
        """
        Returns a list of dicts (one per week) with rich per-week metrics:
          - predicted_rate (mean from RF trees)
          - predicted_risk (std from RF trees, for reporting only)
          - n_launching (projects starting this week)
          - n_active (total active projects this week, GA + context)
          - total_goal (sum of goals for launching projects)
          - fitness_contribution (predicted_rate, before global deviation penalty)
        """
        details = []
        starts_by_week, active_by_week = self._precalculate_schedule_map(individual)

        for t in range(1, self.T + 1):
            current_date = self.start_date + pd.Timedelta(weeks=t-1)
            new_indices = starts_by_week.get(t, [])
            active_indices = active_by_week.get(t, [])

            context_count = len(self._context_cache.get(t, pd.DataFrame()))
            n_active = len(active_indices) + context_count

            row = {
                'week': t,
                'date': current_date,
                'n_launching': len(new_indices),
                'n_active': n_active,
                'predicted_rate': np.nan,
                'predicted_risk': np.nan,
                'fitness_contribution': 0.0,
                'total_goal': 0.0,
            }

            if new_indices:
                active_projects_df = self._build_weekly_dataframe(t, current_date, individual, active_by_week)
                new_projects_df = self.projects.iloc[new_indices].copy()
                state = self.fe.compute_weekly_state(current_date, active_projects_df, new_projects_df)
                rate, risk = self.model.predict_success_distribution(state)
                row['predicted_rate'] = rate
                row['predicted_risk'] = risk
                row['fitness_contribution'] = rate
                if 'goal' in new_projects_df.columns:
                    row['total_goal'] = float(new_projects_df['goal'].sum())

            details.append(row)

        return details

    def get_project_shift_table(self, original_ind, optimized_ind):
        """
        Returns a DataFrame summarizing per-project schedule changes.
        """
        rows = []
        for i, row in self.projects.iterrows():
            orig_w = original_ind[i]
            opt_w = optimized_ind[i]
            rows.append({
                'project_id': row.get('id', i),
                'category': row.get('category', 'N/A'),
                'goal': row.get('goal', 0),
                'duration_days': row.get('duration', 30),
                'original_week': orig_w,
                'optimized_week': opt_w,
                'shift_weeks': opt_w - orig_w,
                'shifted': opt_w != orig_w,
            })
        return pd.DataFrame(rows)

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
        # 1. Real Context (Already running) — use pre-computed cache
        context_subset = self._context_cache.get(t, pd.DataFrame())

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
    def run_pareto(self, args):
        """
        Runs the optimizer across multiple deviation_weight values
        and plots the Pareto front (success rate vs schedule deviation).
        """
        print(f"--- Pareto Sweep Mode (Horizon: {args.weeks} weeks) ---")

        model = SurrogateModel()
        if not model.load():
            print("Error: Model not found. Please run 'python main.py train' first.")
            return

        projects_df, context_df, sim_start_date = self._load_data(args)
        if projects_df.empty:
            return

        original_ind = self._compute_original_individual(projects_df, sim_start_date, args.weeks)

        # Sweep weights from very strict (high weight) to very relaxed (0)
        weights = getattr(args, 'pareto_weights', [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0])
        pareto_points = []

        for w in weights:
            print(f"\n--- Weight = {w} ---")
            problem = CrowdfundingProblem(
                projects_df, model,
                start_date=sim_start_date,
                active_projects=context_df,
                time_horizon=args.weeks,
                deviation_weight=w,
                original_individual=original_ind,
            )

            solver = GeneticSolver(
                problem,
                population_size=args.population,
                generations=args.generations,
            )
            solver.set_initial_individual(original_ind)

            best, fitness, _ = solver.run()
            mean_rate, deviation = problem.decompose_fitness(best)

            print(f"  Fitness={fitness:.4f}  Rate={mean_rate:.4f}  Deviation={deviation:.3f}")
            pareto_points.append({
                'weight': w,
                'mean_rate': mean_rate,
                'deviation': deviation,
                'fitness': fitness,
            })

        # Original schedule metrics
        # Use any problem instance (weight doesn't matter for decompose)
        orig_rate, orig_dev = problem.decompose_fitness(original_ind)
        original_point = {'mean_rate': orig_rate, 'deviation': orig_dev}

        # Sort by deviation for a clean line
        pareto_points.sort(key=lambda p: p['deviation'])

        # Plot
        viz = CrowdfundingVisualizer(projects_df, args.weeks)
        fig = viz.plot_pareto_front(pareto_points, original_point)

        # Save as standalone HTML
        fig.write_html("pareto_front.html", include_plotlyjs='cdn')
        print("\nSaved 'pareto_front.html'")

        # Also save as PNG via matplotlib for quick viewing
        devs = [p['deviation'] for p in pareto_points]
        rates = [p['mean_rate'] for p in pareto_points]
        wts = [p['weight'] for p in pareto_points]

        fig_mpl, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(devs, rates, c=wts, cmap='viridis', s=100, edgecolors='white', zorder=5)
        ax.plot(devs, rates, '--', color='gray', alpha=0.5, zorder=3)
        ax.scatter([orig_dev], [orig_rate], c='red', s=200, marker='*', zorder=6, label='Original')

        for d, r, w in zip(devs, rates, wts):
            ax.annotate(f'w={w}', (d, r), textcoords='offset points',
                        xytext=(5, 8), fontsize=8, alpha=0.8)

        ax.set_xlabel('Mean L1 Deviation (weeks shifted per project)', fontsize=12)
        ax.set_ylabel('Mean Predicted Success Rate', fontsize=12)
        ax.set_title('Pareto Front: Success Rate vs Schedule Deviation', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Deviation Weight')
        fig_mpl.tight_layout()
        fig_mpl.savefig('pareto_front.png', dpi=150)
        print("Saved 'pareto_front.png'")
        plt.close(fig_mpl)

    def _load_data(self, args):
        """Shared data loading logic for run() and run_pareto()."""
        projects_df = pd.DataFrame()
        context_df = pd.DataFrame()
        sim_start_date = None

        if args.projects:
            print(f"Loading real upcoming projects from: {args.projects}")
            try:
                projects_df = pd.read_csv(args.projects)
                if 'Lancé le' in projects_df.columns:
                    projects_df['start_date'] = pd.to_datetime(projects_df['Lancé le'], utc=True)
                elif 'original_start_date' in projects_df.columns:
                    projects_df['start_date'] = pd.to_datetime(projects_df['original_start_date'], utc=True)

                required_map = {'ID projet': 'id', 'Objectif de collecte': 'goal',
                                'Durée de la collecte': 'duration', 'Catégorie principale': 'category'}
                projects_df.rename(columns=required_map, inplace=True)

                if 'goal' in projects_df.columns: projects_df['goal'] = pd.to_numeric(projects_df['goal'], errors='coerce')
                if 'duration' in projects_df.columns: projects_df['duration'] = pd.to_numeric(projects_df['duration'], errors='coerce')

                projects_df = projects_df.dropna(subset=['start_date'])
                if not projects_df.empty:
                    sim_start_date = projects_df['start_date'].min()
                    print(f"Simulation Start Date inferred: {sim_start_date.date()}")
            except Exception as e:
                print(f"Error loading projects file: {e}")
                return pd.DataFrame(), pd.DataFrame(), None

        if args.context:
            print(f"Loading context (active projects) from: {args.context}")
            try:
                context_df = pd.read_csv(args.context)
                if 'Lancé le' in context_df.columns:
                    context_df['start_date'] = pd.to_datetime(context_df['Lancé le'], utc=True)
                elif 'original_start_date' in context_df.columns:
                    context_df['start_date'] = pd.to_datetime(context_df['original_start_date'], utc=True)
                if 'Fini le' in context_df.columns:
                    context_df['end_date'] = pd.to_datetime(context_df['Fini le'], utc=True)
                elif 'original_end_date' in context_df.columns:
                    context_df['end_date'] = pd.to_datetime(context_df['original_end_date'], utc=True)

                required_map = {'ID projet': 'id', 'Objectif de collecte': 'goal',
                                'Durée de la collecte': 'duration', 'Catégorie principale': 'category'}
                context_df.rename(columns=required_map, inplace=True)

                if 'duration' in context_df.columns: context_df['duration'] = pd.to_numeric(context_df['duration'], errors='coerce')
                if 'end_date' not in context_df.columns and 'start_date' in context_df.columns and 'duration' in context_df.columns:
                    context_df['end_date'] = context_df['start_date'] + pd.to_timedelta(context_df['duration'], unit='D')
                context_df = context_df.dropna(subset=['start_date', 'end_date'])
                print(f"Loaded {len(context_df)} active context projects.")
            except Exception as e:
                print(f"Error loading context file: {e}")
                return pd.DataFrame(), pd.DataFrame(), None

        if projects_df.empty:
            print("Error: No projects loaded. You must provide upcoming projects via --projects.")

        return projects_df, context_df, sim_start_date

    @staticmethod
    def _compute_original_individual(projects_df, sim_start_date, weeks):
        original_ind = []
        for _, row in projects_df.iterrows():
            if pd.notna(row['start_date']) and sim_start_date:
                delta_days = (row['start_date'] - sim_start_date).days
                week_idx = math.floor(delta_days / 7) + 1
                week_idx = max(1, min(week_idx, weeks))
                original_ind.append(week_idx)
            else:
                original_ind.append(1)
        return original_ind

    def run(self, args):
        print(f"--- Optimization Mode (Horizon: {args.weeks} weeks) ---")

        model = SurrogateModel()
        if not model.load():
            print("Error: Model not found. Please run 'python main.py train' first.")
            return

        projects_df, context_df, sim_start_date = self._load_data(args)
        if projects_df.empty:
            return

        original_ind = self._compute_original_individual(projects_df, sim_start_date, args.weeks)

        problem = CrowdfundingProblem(
            projects_df,
            model,
            start_date=sim_start_date,
            active_projects=context_df,
            time_horizon=args.weeks,
            deviation_weight=getattr(args, 'deviation_weight', 0.01),
            original_individual=original_ind,
        )

        solver = GeneticSolver(
            problem,
            population_size=args.population,
            generations=args.generations
        )
        solver.set_initial_individual(original_ind)
        
        best, fitness, history = solver.run()
        print(f"Optimization Complete. Best Fitness: {fitness:.4f}")

        # --- Evaluation & Data Collection ---

        original_fitness = problem.evaluate(original_ind)
        print(f"Original Schedule Fitness: {original_fitness:.4f}")

        improvement = ((fitness - original_fitness) / abs(original_fitness) * 100) if original_fitness != 0 else 0
        print(f"Improvement: {improvement:+.1f}%")

        # Collect rich per-week details for both schedules
        orig_details = problem.get_weekly_details(original_ind)
        opt_details = problem.get_weekly_details(best)
        shift_df = problem.get_project_shift_table(original_ind, best)

        print(f"Projects shifted: {shift_df['shifted'].sum()} / {len(shift_df)}")

        # --- Matplotlib Figures ---

        # Figure 1: GA Convergence
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        generations = np.arange(1, len(history) + 1)
        ax1.plot(generations, history, color='#2ecc71', linewidth=2.5, label='Best Fitness')
        ax1.axhline(y=original_fitness, color='#e74c3c', linestyle='--', linewidth=1.5,
                     label=f'Original Fitness ({original_fitness:.4f})')
        ax1.fill_between(generations, original_fitness, history,
                         where=[h > original_fitness for h in history],
                         alpha=0.15, color='#2ecc71', label='Improvement')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness Score', fontsize=12)
        ax1.set_title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
        fig1.tight_layout()
        fig1.savefig('convergence.png', dpi=150)
        print("Saved 'convergence.png'")
        plt.close(fig1)

        # Figure 2: Weekly Success Rate — Optimized (with uncertainty) vs Original
        weeks = np.arange(1, args.weeks + 1)
        opt_rates = np.array([d['predicted_rate'] for d in opt_details])
        opt_stds  = np.array([d['predicted_risk'] for d in opt_details])
        orig_rates = np.array([d['predicted_rate'] for d in orig_details])

        fig2, ax2 = plt.subplots(figsize=(10, 5.5))

        # Optimized — uncertainty band (1-sigma and 2-sigma)
        opt_valid = ~np.isnan(opt_rates)
        w_opt = weeks[opt_valid]
        r_opt = opt_rates[opt_valid]
        s_opt = opt_stds[opt_valid]

        ax2.fill_between(w_opt, r_opt - 2 * s_opt, r_opt + 2 * s_opt,
                         alpha=0.10, color='#2ecc71', label='Optimized $\pm 2\sigma$')
        ax2.fill_between(w_opt, r_opt - s_opt, r_opt + s_opt,
                         alpha=0.25, color='#2ecc71', label='Optimized $\pm 1\sigma$')
        ax2.plot(w_opt, r_opt, 'o-', color='#2ecc71', linewidth=2.5,
                 markersize=8, label='Optimized (predicted)', zorder=5)

        # Original — plain line
        orig_valid = ~np.isnan(orig_rates)
        w_orig = weeks[orig_valid]
        r_orig = orig_rates[orig_valid]
        ax2.plot(w_orig, r_orig, 's--', color='#3498db', linewidth=2,
                 markersize=7, label='Original (predicted)', zorder=4)

        # Annotate weeks with no launches
        for t in weeks:
            if np.isnan(opt_rates[t - 1]) and np.isnan(orig_rates[t - 1]):
                ax2.axvspan(t - 0.4, t + 0.4, alpha=0.07, color='gray')

        ax2.set_xlabel('Week', fontsize=12)
        ax2.set_ylabel('Predicted Success Rate', fontsize=12)
        ax2.set_title('Weekly Success Rate: Optimized (with uncertainty) vs Original',
                       fontsize=14, fontweight='bold')
        ax2.set_xticks(weeks)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        fig2.tight_layout()
        fig2.savefig('weekly_success_rate.png', dpi=150)
        print("Saved 'weekly_success_rate.png'")
        plt.close(fig2)

        # --- Interactive Visualization (Plotly) ---
        try:
            print("Generating Interactive Report...")
            viz = CrowdfundingVisualizer(projects_df, args.weeks)

            figures = []

            # 1. Summary Dashboard (KPI overview)
            figures.append(viz.plot_summary_dashboard(
                original_fitness, fitness, orig_details, opt_details, shift_df))

            # 2. GA Convergence
            figures.append(viz.plot_convergence(history, original_fitness))

            # 3. Weekly Predicted Rate with uncertainty bands
            figures.append(viz.plot_weekly_rate_with_uncertainty(orig_details, opt_details))

            # 4. Per-Week Fitness Contribution (rate - risk)
            figures.append(viz.plot_weekly_fitness_bars(orig_details, opt_details))

            # 5. Risk vs Return scatter
            figures.append(viz.plot_risk_return_scatter(orig_details, opt_details))

            # 6. Platform Load (launching + active projects per week)
            figures.append(viz.plot_weekly_load(orig_details, opt_details))

            # 7. Goal Volume Distribution per week
            figures.append(viz.plot_weekly_goal_distribution(orig_details, opt_details))

            # 8. Gantt chart
            figures.append(viz.compare_schedules_gantt(original_ind, best, sim_start_date))

            # 9. Shift analysis (histogram + by category)
            figures.append(viz.plot_shift_distribution(shift_df))

            viz.save_report(figures, filename="optimization_report.html")
            print("Interactive report saved to 'optimization_report.html'")

        except Exception as e:
            import traceback
            print(f"Error generating interactive report: {e}")
            traceback.print_exc()

