import os
import pandas as pd
from crowdfunding_framework.data_loader import DataLoader

class ExtractionFlow:
    def run(self, args):
        print(f"--- Data Extraction Mode (Ref Date: {args.date}) ---")
        loader = DataLoader()
        projects_df = loader.load_projects()
        
        # Loader already standardizes columns!
        # 'Lancé le' -> 'original_start_date'
        # 'Fini le' -> 'original_end_date_raw'
        
        if 'original_start_date' in projects_df.columns:
             projects_df['start_date'] = pd.to_datetime(projects_df['original_start_date'], utc=True, errors='coerce')
             
        if 'original_end_date_raw' in projects_df.columns:
             projects_df['end_date'] = pd.to_datetime(projects_df['original_end_date_raw'], utc=True, errors='coerce')
        elif 'original_end_date' in projects_df.columns:
             projects_df['end_date'] = pd.to_datetime(projects_df['original_end_date'], utc=True, errors='coerce')
        else:
             # Fallback
             projects_df['end_date'] = projects_df['start_date'] + pd.to_timedelta(projects_df['duration'], unit='D', errors='coerce')
        
        ref_date = pd.to_datetime(args.date, utc=True)
        end_horizon = ref_date + pd.Timedelta(weeks=args.weeks)
        
        # 1. Active Context: Started before Ref, Ends after Ref
        ctx_mask = (projects_df['start_date'] < ref_date) & (projects_df['end_date'] > ref_date)
        context_df = projects_df[ctx_mask].copy()
        
        # 2. Upcoming: Starts in [Ref, Horizon]
        up_mask = (projects_df['start_date'] >= ref_date) & (projects_df['start_date'] <= end_horizon)
        upcoming_df = projects_df[up_mask].copy()
        
        # Save
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        ctx_path = os.path.join(args.output, 'active_context.csv')
        up_path = os.path.join(args.output, 'upcoming_projects.csv')
        
        context_df.to_csv(ctx_path, index=False)
        upcoming_df.to_csv(up_path, index=False)
        
        print(f"Extraction Complete.")
        print(f"Active Context: {len(context_df)} projects -> {ctx_path}")
        print(f"Upcoming Projects: {len(upcoming_df)} projects -> {up_path}")
