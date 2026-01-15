# CEcBaN: Causal Discovery Method Comparison

Synthetic datasets and analysis scripts for comparing causal discovery methods.

## Structure

```
simulations/
├── datasets/          # 4 synthetic time-series datasets
├── results/           # Method outputs and performance metrics
│   └── method_results/  # Individual method outputs per dataset
├── scripts/           # Analysis and visualization scripts
└── *.md              # Documentation files
```

## Datasets

| File | Description | Variables | Size |
|------|-------------|-----------|------|
| `dataset1.csv` | Baseline causal structure | Y1-Y5 | 1000 timesteps |
| `dataset2.csv` | Increased noise | Y1-Y5 | 1000 timesteps |
| `dataset3.csv` | Additional driver (Y0→Y1) | Y0-Y5 | 1000 timesteps |
| `dataset4.csv` | Additional parent (Y4a→Y4) | Y1-Y5, Y4a | 1000 timesteps |

Each dataset includes:
- `.csv` - Time-series data
- `_structure.txt` - Ground truth causal structure
- `_timeseries.png` - Visualization

## Results

### Method Outputs (`results/method_results/`)

Each method was applied to all 4 datasets. Results include detected causal edges:

**CCM-ECCM (Convergent Cross-Mapping):**
- `ccm_dataset1.csv` through `ccm_dataset4.csv`
- Columns: x1, x2, Score, is_Valid, timeToEffect

**PC Algorithm:**
- `pc_dataset1.csv` through `pc_dataset4.csv`
- `pc_all_datasets.csv` (aggregated)
- Columns: x1, x2, lag, correlation, p_value, abs_correlation, method, dataset

**Transfer Entropy:**
- `te_dataset1.csv` through `te_dataset4.csv`
- `te_all_datasets.csv` (aggregated)
- Columns: x1, x2, lag, te_value, method, dataset

**Granger Causality:**
- `granger_dataset1.csv` through `granger_dataset4.csv`
- Columns: source, target, lag, value, p_value, method, dataset

### Performance Metrics (`results/`)

- `metrics.csv` - Detailed per-dataset performance (TP, FP, FN, precision, recall, etc.)
- `summary.csv` - Averaged metrics across all datasets

### Publication Figures (`results/`)

- `panels_AB.pdf/png` - Network size accuracy & false positive control
- `panels_DF.pdf/png` - Edge count accuracy & specificity

## Scripts

### Method Implementation

**`run_ccm.py`** - Convergent Cross-Mapping (CCM/ECCM)
```bash
python scripts/run_ccm.py --file_path datasets/dataset1.csv --output_folder results/
```
Outputs: `CCM_ECCM_curated.csv` (curated causal edges)

**`run_pc_te.py`** - PC Algorithm & Transfer Entropy
```bash
python scripts/run_pc_te.py --file_path datasets/dataset1.csv --output_folder results/
```
Outputs: `pc_results.csv`, `transfer_entropy_results.csv`

**`run_granger.py`** - Granger Causality
```bash
python scripts/run_granger.py
```
Outputs: Granger test results for all datasets

### Results Generation

**`create_metrics.py`** - Calculate performance metrics
```bash
python scripts/create_metrics.py
```
Reads method outputs, calculates TP/FP/FN, generates `metrics.csv` and `summary.csv`

**`create_panels_AB.py`** - Generate figure panels A & B
```bash
python scripts/create_panels_AB.py
```
Outputs: `panels_AB.pdf/png`

**`create_panels_DF.py`** - Generate figure panels D & F
```bash
python scripts/create_panels_DF.py
```
Outputs: `panels_DF.pdf/png`

### Dataset Generation

**`generate_datasets.py`** - Create synthetic datasets
```bash
python scripts/generate_datasets.py
```
Generates all 4 datasets with ground truth structure

## Workflow

```
1. Generate Datasets
   └─ python scripts/generate_datasets.py

2. Run Methods
   ├─ python scripts/run_ccm.py (for each dataset)
   ├─ python scripts/run_pc_te.py (for each dataset)
   └─ python scripts/run_granger.py

3. Calculate Metrics
   └─ python scripts/create_metrics.py

4. Create Figures
   ├─ python scripts/create_panels_AB.py
   └─ python scripts/create_panels_DF.py
```

