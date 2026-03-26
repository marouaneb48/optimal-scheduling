import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class CrowdfundingVisualizer:
    """
    Handles interactive visualizations for the Crowdfunding Optimization results using Plotly.
    """
    
    def __init__(self, projects_df, weeks_horizon):
        self.projects_df = projects_df.copy()
        self.T = weeks_horizon
        
    def compare_schedules_gantt(self, original_starts, optimized_starts, start_date):
        """
        Creates an interactive Gantt chart comparing Original vs Optimized schedules.
        
        Args:
            original_starts (list): List of start week indices (1-based) for original schedule.
            optimized_starts (list): List of start week indices (1-based) for optimized schedule.
            start_date (pd.Timestamp): The base date for week 1.
        """
        fig = go.Figure()
        
        # Helper to calculate dates
        def get_dates(week_idx, duration_days):
            # week_idx is 1-based
            s_date = start_date + pd.Timedelta(weeks=week_idx-1)
            e_date = s_date + pd.Timedelta(days=duration_days)
            return s_date, e_date

        # Add Original Schedule Bars
        for i, row in self.projects_df.iterrows():
            orig_week = original_starts[i]
            s_orig, e_orig = get_dates(orig_week, row.get('duration', 30))
            
            fig.add_trace(go.Bar(
                x=[(e_orig - s_orig).days * 24 * 60 * 60 * 1000], # Duration in ms
                y=[f"Project {row.get('id', i)}: {row.get('category', 'N/A')}"],
                base=s_orig,
                orientation='h',
                name='Original',
                marker=dict(color='lightgray', opacity=0.6),
                hoverinfo='text',
                hovertext=f"Original: Week {orig_week}<br>Start: {s_orig.date()}<br>End: {e_orig.date()}"
            ))

        # Add Optimized Schedule Bars
        for i, row in self.projects_df.iterrows():
            opt_week = optimized_starts[i]
            s_opt, e_opt = get_dates(opt_week, row.get('duration', 30))
            
            color = 'green' if opt_week != original_starts[i] else 'blue'
            
            fig.add_trace(go.Bar(
                x=[(e_opt - s_opt).days * 24 * 60 * 60 * 1000],
                y=[f"Project {row.get('id', i)}: {row.get('category', 'N/A')}"],
                base=s_opt,
                orientation='h',
                name='Optimized',
                marker=dict(color=color, opacity=0.8),
                hoverinfo='text',
                hovertext=f"Optimized: Week {opt_week}<br>Start: {s_opt.date()}<br>End: {e_opt.date()}"
            ))

        fig.update_layout(
            title="Schedule Comparison: Original vs Optimized",
            xaxis_title="Timeline",
            yaxis_title="Projects",
            barmode='overlay', # Overlay to show shifts
            height=max(600, len(self.projects_df) * 30),
            showlegend=True
        )
        
        return fig

    def plot_success_probabilities(self, weekly_probs_opt, weekly_probs_orig_pred, weekly_probs_orig_actual=None):
        """
        Plots the average success probability curve.
        Comparisons:
        1. Optimized (Predicted)
        2. Original (Predicted)
        3. Original (Actual) - Optional
        """
        weeks = list(range(1, self.T + 1))
        
        fig = go.Figure()
        
        # 1. Optimized Schedule (Predicted)
        fig.add_trace(go.Scatter(
            x=weeks, 
            y=weekly_probs_opt,
            mode='lines+markers',
            name='Optimized (Predicted)',
            line=dict(color='green', width=3)
        ))
        
        # 2. Original Schedule (Predicted)
        fig.add_trace(go.Scatter(
            x=weeks, 
            y=weekly_probs_orig_pred,
            mode='lines+markers',
            name='Original (Predicted)',
            line=dict(color='blue', dash='dot')
        ))

        # 3. Original Schedule (Actual) - Optional
        if weekly_probs_orig_actual is not None:
             # handle NaNs (gaps) in actual rate
             clean_y = [y if not np.isnan(y) else None for y in weekly_probs_orig_actual]
             fig.add_trace(go.Scatter(
                x=weeks, 
                y=clean_y,
                mode='lines+markers',
                name='Original (Actual)',
                line=dict(color='gray', dash='dash'),
                opacity=0.6
            ))
        
        fig.update_layout(
            title="Weekly Success Metrics Comparison",
            xaxis_title="Week",
            yaxis_title="Success Probability / Rate",
            yaxis_range=[0, 1.1]
        )
        
        return fig

    def plot_weekly_heatmap(self, weekly_data_matrix):
        """
        Creates a heatmap of weekly metrics (e.g., success probabilities for all projects/weeks).
        
        Args:
            weekly_data_matrix (pd.DataFrame or np.ndarray): 
                Rows = Metrics or Projects
                Cols = Weeks
        """
        fig = px.imshow(
            weekly_data_matrix,
            labels=dict(x="Week", y="Metric", color="Score"),
            x=[f"Week {i+1}" for i in range(len(weekly_data_matrix[0]))],
            aspect="auto",
            title="Weekly Optimization Heatmap"
        )
        return fig

    def save_report(self, figures, filename="optimization_report.html"):
        """
        Saves a list of figures to a single HTML file.
        """
        with open(filename, 'w') as f:
            f.write("<html><head><title>Optimization Report</title></head><body>")
            f.write("<h1>Optimization Results</h1>")
            for fig in figures:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</body></html>")
