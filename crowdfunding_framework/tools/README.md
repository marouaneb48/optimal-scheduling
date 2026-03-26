# Tools Module

This module contains auxiliary utilities for the Crowdfunding Framework, primarily focused on **Data Management**.

## Components

### 1. `extraction_flow.py`
Data Extractor.
-   **Responsibility**: Prepares real data for the Optimization engine.
-   **Logic**:
    -   Loads the massive raw project/contribution database via `data_loader`.
    -   Splits projects relative to a **Reference Date** (Analysis Point):
        1.  **Context**: Projects active on that date (Started before, ends after).
        2.  **Upcoming**: Projects starting within the defined horizon (e.g., next 8 weeks).
    -   Saves these splits as clean CSVs used by `optimization/`.

## Usage

Run the module's entry point to extract data:

```bash
python main.py extract --date <YYYY-MM-DD> --weeks <N>
```

### Arguments
-   `--date`: The reference date to simulate "Now" (e.g., `2024-01-01`).
-   `--weeks`: The horizon to look ahead for upcoming projects.
-   `--output`: Directory to save CSVs (default: current directory).

### Output Files
-   `active_context.csv`: Projects to pass to `--context` in Optimization.
-   `upcoming_projects.csv`: Projects to pass to `--projects` in Optimization.
