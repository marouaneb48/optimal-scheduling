import pandas as pd

import os
import glob
from tqdm import tqdm

class DataLoader:
    def __init__(self, data_dir='data/raw data'):
        self.data_dir = data_dir
        self.projects_dir = os.path.join(data_dir, 'projects')
        self.contributions_dir = os.path.join(data_dir, 'contributions')
        
    def load_projects(self):
        """
        Loads and merges all project CSVs.
        Returns a deduplicated DataFrame with standard columns.
        """
        all_files = glob.glob(os.path.join(self.projects_dir, "**/*.csv"), recursive=True)
        if not all_files:
            print("No project files found.")
            return pd.DataFrame()
            
        print(f"Loading {len(all_files)} project files...")
        df_list = []
        
        # Columns to keep (French names -> Internal use)
        # ID projet, Objectif de collecte, Durée de la collecte, Lancé le, Catégorie principale

        use_cols = ['ID projet', 'Objectif de collecte', 'Durée de la collecte', 
                    'Lancé le', 'Catégorie principale', 'État du projet', "Type d'objectif", 'Fini le']
        
        for f in all_files:
            try:
                # Read header to check cols
                header = pd.read_csv(f, nrows=0).columns
                cols_to_load = [c for c in use_cols if c in header]
                
                df = pd.read_csv(f, usecols=cols_to_load)
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not df_list:
            return pd.DataFrame()
            
        full_df = pd.concat(df_list, ignore_index=True)
        
        # Deduplicate
        if 'ID projet' in full_df.columns:
            full_df.drop_duplicates(subset=['ID projet'], inplace=True)
            
        # --- Apply Notebook Filters ---
        # 1. Status: successful or failed
        # 2. Type: amount
        # 3. Goal > 0
        
        if 'État du projet' in full_df.columns:
            full_df = full_df[full_df['État du projet'].isin(['successful', 'failed'])]
            
        if "Type d'objectif" in full_df.columns:
            full_df = full_df[full_df["Type d'objectif"] == 'amount']
            
        if 'Objectif de collecte' in full_df.columns:
             # Ensure numeric
             full_df['Objectif de collecte'] = pd.to_numeric(full_df['Objectif de collecte'], errors='coerce').fillna(0)
             full_df = full_df[full_df['Objectif de collecte'] > 0]
             
        # Parse Dates
        if 'Lancé le' in full_df.columns:
            full_df['start_date'] = pd.to_datetime(full_df['Lancé le'], errors='coerce', utc=True)
            
        if 'Fini le' in full_df.columns:
            full_df['original_end_date'] = pd.to_datetime(full_df['Fini le'], errors='coerce', utc=True)

        # Rename standard columns
        rename_map = {
            'ID projet': 'id',
            'Objectif de collecte': 'goal',
            'Durée de la collecte': 'duration',
            'Catégorie principale': 'category',
            'État du projet': 'status', # 'successful', 'failed', etc.
            'Lancé le': 'original_start_date',
            'Fini le': 'original_end_date_raw' 
        }
        full_df.rename(columns=rename_map, inplace=True)
        
        # Clean numeric
        full_df['goal'] = pd.to_numeric(full_df['goal'], errors='coerce').fillna(0)
        full_df['duration'] = pd.to_numeric(full_df['duration'], errors='coerce').fillna(30)
        
        print(f"Loaded {len(full_df)} unique projects.")
        return full_df

    def load_contributions_aggregated(self):
        """
        Loads contributions, processes in chunks, and returns aggregated weekly funding per project.
        Returns: DataFrame with ['project_id', 'week_start', 'amount', 'count']
        """
        all_files = glob.glob(os.path.join(self.contributions_dir, "**/*.csv"), recursive=True)
        if not all_files:
            print("No contribution files found.")
            return pd.DataFrame()
            
        print(f"Loading {len(all_files)} contribution files (with chunking)...")
        
        # We accumulate per-project-per-week sums here
        aggregated_chunks = []
        
        seen_ids = set()
        
        all_files.sort() 
        
        use_cols = ['ID contribution', 'ID projet', 'Montant de la contribution', 'Créé le', 'État de la contribution']
        
        chunk_size = 100000 
        
        for f in tqdm(all_files, desc="Processing contribution files"):
            try:
                # Check header
                header = pd.read_csv(f, nrows=0).columns
                cols_to_load = [c for c in use_cols if c in header]
                
                if 'ID contribution' not in cols_to_load:
                    continue # Skip if no ID
                
                for chunk in pd.read_csv(f, usecols=cols_to_load, chunksize=chunk_size):
                    # Filter for OK contributions (Notebook Logic)
                    if 'État de la contribution' in chunk.columns:
                        chunk = chunk[chunk['État de la contribution'] == 'ok']
                        
                    # Filter duplicates
                    chunk = chunk[~chunk['ID contribution'].isin(seen_ids)]
                    
                    if chunk.empty:
                        continue
                        
                    # Add new IDs to seen set
                    seen_ids.update(chunk['ID contribution'].tolist())
                    
                    # Process Chunk
                    # Parse Date -> Week Start
                    chunk['date'] = pd.to_datetime(chunk['Créé le'], errors='coerce', utc=True)
                    chunk.dropna(subset=['date'], inplace=True)
                    chunk['week_start'] = chunk['date'].dt.to_period('W').apply(lambda r: r.start_time)
                    
                    # Rename
                    chunk.rename(columns={
                        'ID projet': 'project_id',
                        'Montant de la contribution': 'amount'
                    }, inplace=True)
                    
                    # Aggregate Chunk: Group by Project + Week
                    # compute sum 'amount' AND count 'ID contribution' (which is just size since we deduplicated)
                    chunk_agg = chunk.groupby(['project_id', 'week_start']).agg(
                        amount=('amount', 'sum'),
                        count=('amount', 'count')
                    ).reset_index()
                    aggregated_chunks.append(chunk_agg)
                    
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
        print("Consolidating aggregations...")
        if not aggregated_chunks:
            return pd.DataFrame()
            
        # Combine all small aggregates
        full_agg = pd.concat(aggregated_chunks, ignore_index=True)
        # Final aggregation (summing sums and counts from different chunks)
        final_df = full_agg.groupby(['project_id', 'week_start']).agg(
            amount=('amount', 'sum'),
            count=('count', 'sum')
        ).reset_index()
        
        # Ensure UTC
        final_df['week_start'] = pd.to_datetime(final_df['week_start'], utc=True)
        
        print(f"Processed contributions. Resulting aggregated records: {len(final_df)}")
        return final_df

if __name__ == "__main__":
    loader = DataLoader() # Testing
    projs = loader.load_projects()
    print("Project Columns:", projs.columns)
