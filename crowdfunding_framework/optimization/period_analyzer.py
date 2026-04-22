import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class PeriodAnalyzer:
    """
    Scans historical weekly data to identify time windows where schedule
    optimization is most likely to produce meaningful gains.

    A good optimization window has:
      - High project congestion (many starting projects competing)
      - High variance in success rate (room to improve)
      - Mixed success rates (not all-success or all-failure weeks)
    """

    def __init__(self, feature_file='weekly_features_from_raw.csv'):
        self.feature_file = feature_file

    def run(self, horizon=8, top_n=10):
        if not os.path.exists(self.feature_file):
            print(f"Error: Feature file '{self.feature_file}' not found.")
            print("Run 'python main.py train --force' first to generate features.")
            return

        df = pd.read_csv(self.feature_file)
        df['week_date'] = pd.to_datetime(df['week_date'], utc=True)
        df = df.sort_values('week_date').reset_index(drop=True)

        print(f"Loaded {len(df)} weeks of data ({df['week_date'].min().date()} to {df['week_date'].max().date()})")
        print(f"Scanning for best {horizon}-week optimization windows...\n")

        # Score each possible window
        windows = []
        for i in range(len(df) - horizon + 1):
            window = df.iloc[i:i + horizon]
            start = window['week_date'].iloc[0]
            end = window['week_date'].iloc[-1]

            # Metrics that indicate optimization potential
            total_starting = window['starting_projects'].sum()
            mean_starting = window['starting_projects'].mean()
            rate_std = window['success_rate'].std()
            rate_mean = window['success_rate'].mean()
            rate_range = window['success_rate'].max() - window['success_rate'].min()

            # Weeks with both successes and failures (not saturated)
            mixed_weeks = ((window['success_rate'] > 0.1) & (window['success_rate'] < 0.9)).sum()

            # Congestion score: total projects × variance × mixed weeks
            # Higher = more potential for schedule reshuffling to help
            score = total_starting * rate_std * (mixed_weeks / horizon)

            windows.append({
                'start_date': start,
                'end_date': end,
                'total_starting': total_starting,
                'mean_starting_per_week': round(mean_starting, 1),
                'success_rate_mean': round(rate_mean, 3),
                'success_rate_std': round(rate_std, 3),
                'success_rate_range': round(rate_range, 3),
                'mixed_weeks': mixed_weeks,
                'score': round(score, 2),
            })

        windows_df = pd.DataFrame(windows).sort_values('score', ascending=False)

        # Print top windows
        print(f"{'='*80}")
        print(f"TOP {top_n} OPTIMIZATION WINDOWS ({horizon}-week horizon)")
        print(f"{'='*80}")
        print(f"{'Rank':<5} {'Start Date':<13} {'End Date':<13} {'Projects':<10} "
              f"{'Avg/Week':<10} {'Rate Mean':<11} {'Rate Std':<10} {'Range':<8} {'Score':<8}")
        print(f"{'-'*80}")

        for rank, (_, row) in enumerate(windows_df.head(top_n).iterrows(), 1):
            print(f"{rank:<5} {str(row['start_date'].date()):<13} {str(row['end_date'].date()):<13} "
                  f"{row['total_starting']:<10} {row['mean_starting_per_week']:<10} "
                  f"{row['success_rate_mean']:<11} {row['success_rate_std']:<10} "
                  f"{row['success_rate_range']:<8} {row['score']:<8}")

        # Recommendation
        best = windows_df.iloc[0]
        print(f"\n{'='*80}")
        print(f"RECOMMENDATION")
        print(f"{'='*80}")
        print(f"Best period to optimize: {best['start_date'].date()} to {best['end_date'].date()}")
        print(f"  - {int(best['total_starting'])} projects launching across {horizon} weeks")
        print(f"  - Success rate varies from "
              f"{best['success_rate_mean'] - best['success_rate_range']/2:.1%} to "
              f"{best['success_rate_mean'] + best['success_rate_range']/2:.1%}")
        print(f"  - High congestion + high variance = most room for improvement")
        print(f"\nTo optimize this period:")
        print(f"  python main.py extract --date {best['start_date'].date()} --weeks {horizon}")
        print(f"  python main.py optimize --projects upcoming_projects.csv --context active_projects.csv --weeks {horizon}")

        # Plot: congestion + success rate over time with top windows highlighted
        self._plot_analysis(df, windows_df, horizon, top_n)

    def _plot_analysis(self, df, windows_df, horizon, top_n):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        dates = df['week_date']

        # Panel 1: Project congestion
        ax1 = axes[0]
        ax1.fill_between(dates, df['starting_projects'], alpha=0.3, color='#3498db', label='Starting')
        ax1.plot(dates, df['starting_projects'], color='#3498db', linewidth=1)
        if 'current_projects' in df.columns:
            ax1.plot(dates, df['current_projects'], color='#2ecc71', linewidth=1, alpha=0.7, label='Active')
        ax1.set_ylabel('Projects', fontsize=11)
        ax1.set_title('Platform Congestion Over Time', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Success rate
        ax2 = axes[1]
        ax2.plot(dates, df['success_rate'], color='#e67e22', linewidth=1.2)
        ax2.fill_between(dates, df['success_rate'], alpha=0.2, color='#e67e22')
        ax2.set_ylabel('Success Rate', fontsize=11)
        ax2.set_title('Weekly Success Rate', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Optimization score (rolling)
        ax3 = axes[2]
        scores = windows_df.sort_values('start_date')
        ax3.fill_between(scores['start_date'], scores['score'], alpha=0.3, color='#9b59b6')
        ax3.plot(scores['start_date'], scores['score'], color='#9b59b6', linewidth=1.2)
        ax3.set_ylabel('Optimization Score', fontsize=11)
        ax3.set_title(f'Optimization Potential ({horizon}-week rolling window)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)

        # Highlight top windows on all panels
        top = windows_df.head(min(3, top_n))
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        for idx, (_, row) in enumerate(top.iterrows()):
            for ax in axes:
                ax.axvspan(row['start_date'], row['end_date'],
                           alpha=0.12, color=colors[idx % len(colors)],
                           label=f"#{idx+1}" if ax == axes[0] else None)

        fig.tight_layout()
        fig.savefig('period_analysis.png', dpi=150)
        print("\nSaved 'period_analysis.png'")
        plt.close(fig)
