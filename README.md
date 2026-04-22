# Crowdfunding Optimization Framework

## Overview
This project optimizes the launch schedule of crowdfunding projects to maximize their success probability, considering weekly platform congestion, seasonality, and project diversity.

## Structure
- `crowdfunding_framework/`: Core package.
    - `main.py`: **Unified CLI** for all project tasks.
    - `data_loader.py`: Loads raw project and contribution CSVs.
    - `modeling/`: Feature engineering and Surrogate Model (Random Forest).
    - `optimization/`: Genetic Algorithm solver and Interactive Visualization.
- `data/raw data/`: Place your Ulule data exports here.

## Standard Workflow: Summer 2023 Simulation

Run these commands from the project root to perform the full optimization pipeline relative to the Summer 2023 period.

### 1. Train the Model
Generates features and trains the model.
**Important:** We restrict training data to *before* the simulation starts (July 1st, 2023) to avoid data leakage.
```bash
python crowdfunding_framework/main.py train --force --end-date 2023-07-01
```

### 2. Extract Data (Data Preparation)
Extracts the context and upcoming projects for the simulation window (Starting July 1st, 2023).
```bash
python crowdfunding_framework/main.py extract \
  --date "2023-07-01" \
  --weeks 8 \
  --output simulation_input
```
*Creates `simulation_input/upcoming_projects.csv` and `simulation_input/active_context.csv`.*

### 3. Run Optimization
Optimizes the schedule using the extracted data.
```bash
python crowdfunding_framework/main.py optimize \
  --projects "simulation_input/upcoming_projects.csv" \
  --context "simulation_input/active_context.csv" \
  --weeks 8 \
  --population 50 \
  --generations 30
```

## Outputs
- **`optimization_report.html`**: Interactive report with Gantt charts and Heatmaps.
- **`optimization_results.png`**: Static summary plots.


## Pareto

# Default sweep (9 weights from 0 to 1.0)
python -m crowdfunding_framework.main pareto --projects simulation_input/upcoming_projects.csv --context simulation_input/active_context.csv

# Custom weights
python -m crowdfunding_framework.main pareto --projects simulation_input/upcoming_projects.csv --context simulation_input/active_context.csv --weights 0 0.01 0.02 0.03 0.05 0.06 --population 50 --generations 100

