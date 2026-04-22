import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class CrowdfundingVisualizer:
    """
    Interactive visualizations for Crowdfunding Scheduling Optimization.
    Designed for a continuous regression target (predicted success rate).
    """

    def __init__(self, projects_df, weeks_horizon):
        self.projects_df = projects_df.copy()
        self.T = weeks_horizon

    # ──────────────────────────────────────────────
    # 1. GANTT — Original vs Optimized schedule
    # ──────────────────────────────────────────────
    def compare_schedules_gantt(self, original_starts, optimized_starts, start_date):
        """
        Overlaid Gantt chart: gray = original, colored = optimized.
        Projects that moved are highlighted with arrows.
        """
        fig = go.Figure()

        def get_dates(week_idx, duration_days):
            s = start_date + pd.Timedelta(weeks=week_idx - 1)
            e = s + pd.Timedelta(days=duration_days)
            return s, e

        labels = []
        for i, row in self.projects_df.iterrows():
            pid = row.get('id', i)
            cat = row.get('category', 'N/A')
            labels.append(f"P{pid} ({cat})")

        # Original bars (background)
        for i, row in self.projects_df.iterrows():
            s, e = get_dates(original_starts[i], row.get('duration', 30))
            fig.add_trace(go.Bar(
                x=[(e - s).days * 86400000], y=[labels[i]], base=s,
                orientation='h', name='Original' if i == 0 else None,
                marker=dict(color='lightgray', opacity=0.5),
                showlegend=(i == 0), legendgroup='original',
                hovertext=f"Original W{original_starts[i]}<br>{s.date()} - {e.date()}",
                hoverinfo='text',
            ))

        # Optimized bars (foreground)
        for i, row in self.projects_df.iterrows():
            s, e = get_dates(optimized_starts[i], row.get('duration', 30))
            shifted = optimized_starts[i] != original_starts[i]
            color = '#2ecc71' if shifted else '#3498db'
            fig.add_trace(go.Bar(
                x=[(e - s).days * 86400000], y=[labels[i]], base=s,
                orientation='h',
                name=('Optimized (shifted)' if shifted else 'Optimized (kept)') if i == 0 else None,
                marker=dict(color=color, opacity=0.85,
                            line=dict(width=1.5, color='white') if shifted else dict(width=0)),
                showlegend=(i == 0), legendgroup='optimized',
                hovertext=f"Optimized W{optimized_starts[i]}<br>{s.date()} - {e.date()}"
                          + (f"<br>Shift: {optimized_starts[i] - original_starts[i]:+d}w" if shifted else ""),
                hoverinfo='text',
            ))

        fig.update_layout(
            title="Schedule Comparison: Original vs Optimized",
            xaxis_title="Timeline", yaxis_title="Projects",
            barmode='overlay',
            height=max(600, len(self.projects_df) * 35),
            showlegend=True,
            template='plotly_white',
        )
        return fig

    # ──────────────────────────────────────────────
    # 2. WEEKLY PREDICTED SUCCESS RATE — with uncertainty band
    # ──────────────────────────────────────────────
    def plot_weekly_rate_with_uncertainty(self, orig_details, opt_details):
        """
        Line chart of predicted success rate per week (mean +/- std)
        for both original and optimized schedules.
        """
        fig = go.Figure()
        weeks = list(range(1, self.T + 1))

        for details, name, color, dash in [
            (opt_details, 'Optimized', '#2ecc71', 'solid'),
            (orig_details, 'Original', '#3498db', 'dot'),
        ]:
            rates = [d['predicted_rate'] for d in details]
            risks = [d['predicted_risk'] for d in details]

            # Replace NaN with None for plotly gap handling
            y_mean = [r if not np.isnan(r) else None for r in rates]
            y_upper = [(r + s) if (not np.isnan(r) and not np.isnan(s)) else None for r, s in zip(rates, risks)]
            y_lower = [(r - s) if (not np.isnan(r) and not np.isnan(s)) else None for r, s in zip(rates, risks)]

            # Uncertainty band
            fig.add_trace(go.Scatter(
                x=weeks + weeks[::-1],
                y=[u if u is not None else None for u in y_upper] +
                  [l if l is not None else None for l in y_lower[::-1]],
                fill='toself', fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in color else f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)',
                line=dict(width=0), showlegend=False, name=f'{name} band',
                hoverinfo='skip',
            ))

            # Mean line
            fig.add_trace(go.Scatter(
                x=weeks, y=y_mean,
                mode='lines+markers', name=name,
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=8),
                hovertemplate='Week %{x}<br>Rate: %{y:.4f}<extra></extra>',
            ))

        fig.update_layout(
            title="Predicted Weekly Success Rate (mean +/- 1 std)",
            xaxis_title="Week", yaxis_title="Predicted Success Rate",
            template='plotly_white', hovermode='x unified',
        )
        return fig

    # ──────────────────────────────────────────────
    # 3. WEEKLY FITNESS CONTRIBUTION — bar chart
    # ──────────────────────────────────────────────
    def plot_weekly_fitness_bars(self, orig_details, opt_details):
        """
        Grouped bar chart showing per-week fitness contribution
        (predicted rate) for original vs optimized.
        """
        weeks = list(range(1, self.T + 1))
        orig_fit = [d['fitness_contribution'] for d in orig_details]
        opt_fit = [d['fitness_contribution'] for d in opt_details]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weeks, y=orig_fit, name='Original',
            marker_color='#3498db', opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=weeks, y=opt_fit, name='Optimized',
            marker_color='#2ecc71', opacity=0.9,
        ))
        fig.update_layout(
            title="Per-Week Fitness Contribution (Predicted Rate)",
            xaxis_title="Week", yaxis_title="Fitness Contribution",
            barmode='group', template='plotly_white',
        )
        return fig

    # ──────────────────────────────────────────────
    # 4. PLATFORM LOAD — projects launching & active per week
    # ──────────────────────────────────────────────
    def plot_weekly_load(self, orig_details, opt_details):
        """
        Stacked area / bar showing how many projects are launching
        and active each week, comparing both schedules.
        """
        weeks = list(range(1, self.T + 1))
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Projects Launching per Week",
                                            "Total Active Projects per Week"],
                            shared_yaxes=False)

        # Launching
        fig.add_trace(go.Bar(x=weeks, y=[d['n_launching'] for d in orig_details],
                             name='Original (launching)', marker_color='#3498db', opacity=0.6),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=weeks, y=[d['n_launching'] for d in opt_details],
                             name='Optimized (launching)', marker_color='#2ecc71', opacity=0.8),
                      row=1, col=1)

        # Active
        fig.add_trace(go.Scatter(x=weeks, y=[d['n_active'] for d in orig_details],
                                 mode='lines+markers', name='Original (active)',
                                 line=dict(color='#3498db', dash='dot')),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=weeks, y=[d['n_active'] for d in opt_details],
                                 mode='lines+markers', name='Optimized (active)',
                                 line=dict(color='#2ecc71', width=3)),
                      row=1, col=2)

        fig.update_layout(
            title="Platform Load: Scheduling Distribution",
            template='plotly_white', height=450,
            barmode='group',
        )
        fig.update_xaxes(title_text="Week", row=1, col=1)
        fig.update_xaxes(title_text="Week", row=1, col=2)
        fig.update_yaxes(title_text="# Projects", row=1, col=1)
        fig.update_yaxes(title_text="# Projects", row=1, col=2)
        return fig

    # ──────────────────────────────────────────────
    # 5. RISK vs RETURN SCATTER — per-week trade-off
    # ──────────────────────────────────────────────
    def plot_risk_return_scatter(self, orig_details, opt_details):
        """
        Scatter plot: X = risk (std), Y = rate (mean) for each week.
        Visualizes the risk/return trade-off achieved by optimization.
        """
        fig = go.Figure()

        for details, name, color, symbol in [
            (orig_details, 'Original', '#3498db', 'circle'),
            (opt_details, 'Optimized', '#2ecc71', 'diamond'),
        ]:
            rates = [d['predicted_rate'] for d in details if not np.isnan(d['predicted_rate'])]
            risks = [d['predicted_risk'] for d in details if not np.isnan(d['predicted_rate'])]
            wks = [d['week'] for d in details if not np.isnan(d['predicted_rate'])]

            fig.add_trace(go.Scatter(
                x=risks, y=rates,
                mode='markers+text', name=name,
                marker=dict(color=color, size=14, symbol=symbol,
                            line=dict(width=1, color='white')),
                text=[f"W{w}" for w in wks],
                textposition='top center',
                hovertemplate='Week %{text}<br>Risk (std): %{x:.4f}<br>Rate: %{y:.4f}<extra></extra>',
            ))

        fig.update_layout(
            title="Risk-Return Trade-off per Week",
            xaxis_title="Prediction Uncertainty (std)",
            yaxis_title="Predicted Success Rate",
            template='plotly_white',
        )
        return fig

    # ──────────────────────────────────────────────
    # 6. PROJECT SHIFT ANALYSIS — waterfall / distribution
    # ──────────────────────────────────────────────
    def plot_shift_distribution(self, shift_df):
        """
        Histogram of schedule shifts + table of top movers.
        shift_df: DataFrame from problem.get_project_shift_table()
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Distribution of Week Shifts", "Shifts by Category"],
            specs=[[{"type": "histogram"}, {"type": "bar"}]],
        )

        shifts = shift_df['shift_weeks']
        fig.add_trace(go.Histogram(
            x=shifts, nbinsx=2 * self.T + 1,
            marker_color='#2ecc71', opacity=0.8,
            name='Shift distribution',
        ), row=1, col=1)

        # Average shift by category
        cat_shifts = shift_df.groupby('category')['shift_weeks'].mean().sort_values()
        fig.add_trace(go.Bar(
            x=cat_shifts.values, y=cat_shifts.index,
            orientation='h', marker_color='#e67e22', name='Avg shift by category',
        ), row=1, col=2)

        fig.update_layout(
            title="Schedule Shift Analysis",
            template='plotly_white', height=450,
            showlegend=False,
        )
        fig.update_xaxes(title_text="Shift (weeks)", row=1, col=1)
        fig.update_xaxes(title_text="Avg Shift (weeks)", row=1, col=2)
        return fig

    # ──────────────────────────────────────────────
    # 7. GA CONVERGENCE — interactive version
    # ──────────────────────────────────────────────
    def plot_convergence(self, history, original_fitness):
        """
        GA convergence curve with baseline reference line.
        """
        gens = list(range(1, len(history) + 1))
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=gens, y=history,
            mode='lines', name='Best Fitness',
            line=dict(color='#2ecc71', width=3),
            fill='tozeroy', fillcolor='rgba(46,204,113,0.1)',
        ))

        # Baseline
        fig.add_hline(
            y=original_fitness, line_dash='dash', line_color='#e74c3c',
            annotation_text=f'Original: {original_fitness:.4f}',
            annotation_position='top left',
        )

        fig.update_layout(
            title="Genetic Algorithm Convergence",
            xaxis_title="Generation", yaxis_title="Fitness Score",
            template='plotly_white',
        )
        return fig

    # ──────────────────────────────────────────────
    # 8. SUMMARY DASHBOARD — KPI cards + comparison
    # ──────────────────────────────────────────────
    def plot_summary_dashboard(self, original_fitness, optimized_fitness,
                               orig_details, opt_details, shift_df):
        """
        A single-figure dashboard with key metrics.
        """
        improvement_pct = ((optimized_fitness - original_fitness) / abs(original_fitness) * 100) if original_fitness != 0 else 0

        orig_mean_rate = np.nanmean([d['predicted_rate'] for d in orig_details])
        opt_mean_rate = np.nanmean([d['predicted_rate'] for d in opt_details])
        orig_mean_risk = np.nanmean([d['predicted_risk'] for d in orig_details])
        opt_mean_risk = np.nanmean([d['predicted_risk'] for d in opt_details])
        n_shifted = int(shift_df['shifted'].sum())
        n_total = len(shift_df)

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Total Fitness", "Avg Predicted Rate", "Avg Uncertainty (Risk)",
                "Projects Shifted", "Rate Improvement", "Risk Reduction",
            ],
            specs=[[{"type": "bar"}] * 3, [{"type": "bar"}] * 3],
            vertical_spacing=0.25,
        )

        labels = ['Original', 'Optimized']
        colors = ['#3498db', '#2ecc71']

        pairs = [
            (1, 1, [original_fitness, optimized_fitness], "Score"),
            (1, 2, [orig_mean_rate, opt_mean_rate], "Rate"),
            (1, 3, [orig_mean_risk, opt_mean_risk], "Std"),
        ]
        for r, c, vals, ylab in pairs:
            fig.add_trace(go.Bar(x=labels, y=vals, marker_color=colors,
                                 text=[f"{v:.4f}" for v in vals], textposition='outside',
                                 showlegend=False), row=r, col=c)
            fig.update_yaxes(title_text=ylab, row=r, col=c)

        # Projects shifted
        fig.add_trace(go.Bar(
            x=['Shifted', 'Unchanged'],
            y=[n_shifted, n_total - n_shifted],
            marker_color=['#e67e22', '#bdc3c7'],
            text=[n_shifted, n_total - n_shifted], textposition='outside',
            showlegend=False,
        ), row=2, col=1)

        # Rate improvement
        rate_change = opt_mean_rate - orig_mean_rate
        fig.add_trace(go.Bar(
            x=['Rate Change'], y=[rate_change],
            marker_color='#2ecc71' if rate_change > 0 else '#e74c3c',
            text=[f"{rate_change:+.4f}"], textposition='outside',
            showlegend=False,
        ), row=2, col=2)

        # Risk reduction
        risk_change = opt_mean_risk - orig_mean_risk
        fig.add_trace(go.Bar(
            x=['Risk Change'], y=[risk_change],
            marker_color='#2ecc71' if risk_change < 0 else '#e74c3c',
            text=[f"{risk_change:+.4f}"], textposition='outside',
            showlegend=False,
        ), row=2, col=3)

        fig.update_layout(
            title=f"Optimization Summary — Fitness Improvement: {improvement_pct:+.1f}%",
            template='plotly_white', height=600,
            showlegend=False,
        )
        return fig

    # ──────────────────────────────────────────────
    # 9. GOAL DISTRIBUTION PER WEEK
    # ──────────────────────────────────────────────
    def plot_weekly_goal_distribution(self, orig_details, opt_details):
        """
        Bar chart of total fundraising goal volume launched per week.
        Helps visualize whether the optimizer spreads out monetary competition.
        """
        weeks = list(range(1, self.T + 1))
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=weeks, y=[d['total_goal'] for d in orig_details],
            name='Original', marker_color='#3498db', opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            x=weeks, y=[d['total_goal'] for d in opt_details],
            name='Optimized', marker_color='#2ecc71', opacity=0.85,
        ))

        fig.update_layout(
            title="Total Fundraising Goal Volume Launched per Week",
            xaxis_title="Week", yaxis_title="Total Goal (EUR)",
            barmode='group', template='plotly_white',
        )
        return fig

    # ──────────────────────────────────────────────
    # 10. PARETO FRONT — Success Rate vs Deviation
    # ──────────────────────────────────────────────
    def plot_pareto_front(self, pareto_points, original_point):
        """
        Scatter plot of (mean_L1_deviation, mean_success_rate) for each
        deviation_weight tested. Visualizes the tradeoff between schedule
        disruption and predicted performance.

        pareto_points: list of dicts with keys:
            'weight', 'mean_rate', 'deviation', 'fitness'
        original_point: dict with 'mean_rate', 'deviation' for the original schedule
        """
        weights = [p['weight'] for p in pareto_points]
        rates = [p['mean_rate'] for p in pareto_points]
        devs = [p['deviation'] for p in pareto_points]

        fig = go.Figure()

        # Pareto points
        fig.add_trace(go.Scatter(
            x=devs, y=rates,
            mode='lines+markers+text',
            marker=dict(size=12, color=weights, colorscale='Viridis',
                        colorbar=dict(title='Deviation<br>Weight'),
                        line=dict(width=1, color='white')),
            text=[f"w={w}" for w in weights],
            textposition='top center',
            textfont=dict(size=9),
            name='Pareto Front',
            hovertemplate=(
                'Deviation: %{x:.3f}<br>'
                'Success Rate: %{y:.4f}<br>'
                'Weight: %{text}<extra></extra>'
            ),
        ))

        # Original schedule point
        fig.add_trace(go.Scatter(
            x=[original_point['deviation']],
            y=[original_point['mean_rate']],
            mode='markers+text',
            marker=dict(size=16, color='#e74c3c', symbol='star'),
            text=['Original'],
            textposition='bottom center',
            name='Original Schedule',
        ))

        fig.update_layout(
            title="Pareto Front: Success Rate vs Schedule Deviation",
            xaxis_title="Mean L1 Deviation (weeks shifted per project)",
            yaxis_title="Mean Predicted Success Rate",
            template='plotly_white',
            hovermode='closest',
        )
        return fig

    # ──────────────────────────────────────────────
    # SAVE REPORT
    # ──────────────────────────────────────────────
    def save_report(self, figures, filename="optimization_report.html"):
        """Saves a list of Plotly figures to a single styled HTML file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Crowdfunding Optimization Report</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f8f9fa; }
  h1 { color: #2c3e50; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }
  .chart-container { background: white; border-radius: 8px; padding: 15px;
                     margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  hr { border: none; border-top: 1px solid #ecf0f1; margin: 30px 0; }
</style>
</head><body>
<h1>Crowdfunding Schedule Optimization Report</h1>
""")
            for fig in figures:
                f.write('<div class="chart-container">\n')
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write('\n</div><hr>\n')
            f.write("</body></html>")
