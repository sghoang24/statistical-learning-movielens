# Statistical Learning Project - MovieLens (ml-25m).

This guide walks you through setting up your environment, cloning the source code,
and preparing the data to run the project.

## 1. Prerequisites

- **Git**: to clone the source code.
- **Python 3.12+**: to run the scripts in `src/`.
- **~1 GB of free disk space**: the `ml-25m.zip` file is about 250 MB and extracts to roughly 1 GB.

## 2. Install Git

### Windows

1. Download the installer from: <https://git-scm.com/download/win>
2. Run the `.exe` file and keep the default options.
3. Verify the installation in PowerShell:

```powershell
git --version
```

### macOS

```bash
brew install git
git --version
```

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y git
git --version
```

## 3. Clone the source code

```bash
git clone https://github.com/sghoang24/statistical-learning-movielens.git
cd statistical-learning-movielens
```

## 4. Create a Python environment and install dependencies

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 5. Download the MovieLens ml-25m dataset

This project uses the **MovieLens 25M** dataset from GroupLens.

- Direct download link: <https://files.grouplens.org/datasets/movielens/ml-25m.zip>
- Dataset description page: <https://grouplens.org/datasets/movielens/25m/>

### Option A - Download via the browser

Open the link above and download the `ml-25m.zip` file.

### Option B - Download via the command line

```powershell
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-25m.zip" -OutFile "ml-25m.zip"
```

```bash
# macOS / Linux
curl -L -o ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
```

## 6. Extract into `data/raw`

The goal is to place the CSV files at exactly this location:

```
data/raw/ml-25m/
├── genome-scores.csv
├── genome-tags.csv
├── links.csv
├── movies.csv
├── ratings.csv
├── tags.csv
└── README.txt
```

> Note: `ml-25m.zip` already contains a top-level folder named `ml-25m`. So extract into
> `data/raw` to avoid creating an extra `ml-25m/ml-25m` layer. The scripts in `src/ratings.py`
> read the data from `data/raw/ml-25m/ratings.csv`.

Run these from the project root:

### Windows (PowerShell)

```powershell
# Create the destination folder
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null

# Extract (avoid Expand-Archive, which is slow for large files)
tar -xf ml-25m.zip -C data\raw
```

If `tar` is not available, you can use `Expand-Archive`:

```powershell
Expand-Archive -Path "ml-25m.zip" -DestinationPath "data\raw" -Force
```

### macOS / Linux

```bash
mkdir -p data/raw
unzip -q ml-25m.zip -d data/raw
```

### Verify the result

```bash
ls data/raw/ml-25m
```

You should see `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `genome-scores.csv`,
`genome-tags.csv`, and `README.txt`.

Clean up the zip file after extracting (optional):

```powershell
# Windows
Remove-Item ml-25m.zip
```

```bash
# macOS / Linux
rm ml-25m.zip
```

## 7. Run the pipeline

Run every command from the project root with the virtual environment active.

### Phase 1: prepare the datasets

```bash
# Writes the catalogue, validation positives, phase mappings and a checksum
# manifest into data/processed/
python -m src.prepare_datasets

# Optional: popularity baseline on the 2017 validation set (Phase 1 reference)
python -m src.run_popularity_validation
```

To check your datasets match the team's, compare the hashes in
`data/processed/dataset_manifest.json` with
`deliverables/reference_dataset_manifest.json`.

### Phase 2: standalone models

Run these in order. Tuning must come before the benchmark, because the
benchmark reads the settings that tuning selects.

```bash
# 1. Synthetic self-checks (seconds; no MovieLens data needed)
python -m src.check_phase2

# 2. Base tuning grid on 2017 validation
#    -> outputs/phase2_tuning_grid.csv, outputs/phase2_selected_config.json
python -m src.tune_models

# 3. Extension probes, combined with the base grid
#    -> outputs/phase2_tuning_extension.csv, outputs/phase2_tuning_grid_extended.csv,
#       outputs/phase2_selected_config_extended.json  (the current selection)
python -m src.tune_extension

# 4. Full validation benchmark for all four models, using the extended selection
#    -> outputs/phase2_validation_*.csv and outputs/phase2_validation_environment.json
python -m src.run_phase2_validation

# 5. Independent check of the saved ranking metrics (read-only, ~15 seconds)
python -m src.check_phase2_metrics
```

Notes:

- **Tuning is slow and usually doesn't need repeating.** Step 2 refits every
  setting in the base grid. Step 3 runs only the extension settings not
  already recorded. If you change model code or the fallback, rerun both so the
  grid compares like with like.
- **The benchmark takes several minutes.** Item-CF and matrix factorization
  each take about four minutes to fit. The recorded run completed on a 16 GB machine; peak memory was not measured.
- **The benchmark replaces its previous outputs.** It deletes its own earlier
  `phase2_validation_*` files before fitting, so a failed run can't leave old
  results behind. Commit or copy them first if you need them. Its environment
  file records the settings, fallback, git commit and whether the code had
  uncommitted changes.
- **Settings come from a file.** The benchmark reads
  `outputs/phase2_selected_config_extended.json` and stops with an error if it
  is missing or incomplete. Missing scores are filled by a separate bias model
  (λ = 10/10), the same one tuning uses.
- **Ranking scores are never clipped.** RMSE/MAE use predictions clipped to the
  MovieLens rating range [0.5, 5].

### Phase 2 notebook

`notebook/Phase2_Standalone_Models.ipynb` explains each model and reproduces the
benchmark step by step, using the same selection file and fallback as the
runner. Its final cell compares its results with
`outputs/phase2_validation_summary.csv`. Select the `.venv` kernel and run all
cells; it needs the processed datasets and the tuning outputs above.

Results and method details are written up in
`deliverables/MovieLens_P2_Standalone_Benchmarks.md` and
`deliverables/MovieLens_P2_Implementation_Summary.md`.

## Project structure

```
data/raw/          # Raw MovieLens data (not committed to git)
data/processed/    # Datasets generated by src.prepare_datasets (not committed to git)
deliverables/      # Per-phase documents, plan and reports
notebook/          # Phase 2 walkthrough notebook
outputs/           # Tuning grids, selected settings and validation results (committed)
src/               # Data processing, models, tuning, evaluation and checks
requirements.txt   # List of dependencies
```

> `data/raw/` and `data/processed/` are listed in `.gitignore`, so the downloaded
> and generated data will not be pushed to git. Everyone regenerates
> `data/processed/` with `python -m src.prepare_datasets`.
