
# CEcBaN: CCM ECCM Bayesian Network

CEcBaN is a comprehensive causal analysis platform that implements the CCM-ECCM methodology for identifying causal relationships in time series data and constructing Bayesian networks for probabilistic modeling and prediction.

## Overview

CEcBaN provides an integrated workflow for:
- Convergent Cross Mapping (CCM) analysis
- Extended Convergent Cross Mapping (ECCM) analysis  
- Surrogate data testing for statistical validation
- Bayesian network construction and inference
- AI-powered literature-based interaction discovery

The platform is based on the methodology described in Tal et al. (2024) and provides both a user-friendly web interface and command-line tools for advanced users.


## Installation

### Prerequisites
- Linux/Unix system (Ubuntu/Debian recommended)
- Python 3.9
- Git

### Quick Installation

1. Clone the repository:
```bash
git clone https://github.com/ot483/cecban.git
cd cecban
```

2. Run the installation script:
```bash
chmod +x install.sh
./install.sh
```

3. (Optional) Install LLM packages for AI features:
```bash
chmod +x install_llm.sh
./install_llm.sh
```

### Manual Installation

If the automatic installation fails, you can install dependencies manually:

```bash
# Create virtual environment
python3.9 -m venv cecban_env
source cecban_env/bin/activate

# Install core packages
pip install numpy==1.24.4 scipy==1.10.1 pandas==1.4.3
pip install matplotlib==3.5.2 seaborn==0.11.2 scikit-learn==1.1.1
pip install networkx==2.8.8 dash==2.9.2 plotly==5.14.0
pip install bnlearn==0.7.13 pyEDM==1.13.1.0

# Install LLM packages (optional)
pip install openai>=1.0.0 google-generativeai>=0.3.0
```

## Usage

### Starting the Application

```bash
chmod +x run.sh
./run.sh
```

The application will start on `http://127.0.0.1:8050`

### Basic Workflow

1. **Data Upload**: Upload a CSV file with time series data
2. **Configuration**: Set analysis parameters and target variables
3. **CCM-ECCM Analysis**: Run initial causal discovery
4. **Network Refinement**: Review and edit discovered relationships
5. **Surrogate Analysis**: Validate relationships with surrogate testing
6. **Bayesian Network**: Build probabilistic model
7. **Inference**: Make predictions and explore scenarios

### Data Format

Input data should be a CSV file with:
- First column: Date/time stamps (named 'Date' or 'date')
- Subsequent columns: Time series variables
- No missing values
- Sufficient data points (recommended: 40+ observations)
- No '_' in column names

Example:
```csv
Date,Temperature,ChlorophyllA,Nitrate,Phosphate
2020-01-01,15.2,2.1,1.5,0.3
2020-01-08,16.1,2.3,1.4,0.4
2020-01-15,17.3,2.5,1.3,0.5
```

### AI-Powered Interaction Discovery

To use the LLM features:

1. Obtain API keys from:
   - OpenAI: https://platform.openai.com/api-keys
   - Google AI: https://makersuite.google.com/app/apikey

2. Set environment variables (optional):
```bash
export OPENAI_API_KEY='your-openai-key'
export GOOGLE_API_KEY='your-google-key'
```

3. Or enter API keys directly in the web interface


## Parameters Guide

### Core Parameters
- **Target Column**: Primary variable for prediction
- **Confounders**: External driving variables to exclude from causal discovery
- **Subset Length**: Window size for analysis (30-100 recommended)
- **Embedding Dimension**: Phase space reconstruction dimension (2-7 typical)
- **Resample Frequency**: Time resolution (e.g., '1D', '1W', '1M')

### Advanced Parameters
- **ECCM Window Size**: Window for lag optimization
- **Convergence Method**: Algorithm for detecting CCM convergence
- **Surrogate Testing**: Number of surrogate datasets for validation
- **Bayesian Network**: Training fraction and probability cutoffs

## Output Files

The analysis generates several output files in the results directory:

### Analysis Results
- `CCM_ECCM_curated.csv`: Validated causal relationships
- `network_plot.png`: Network visualization
- `Surr_plot.png`: Surrogate analysis results

### Bayesian Network Results
- `bnlearn_model.pkl`: Trained Bayesian network model
- `CausalDAG_NET.png`: Final causal network
- `BN_model_validation.png`: Model performance metrics
- `scenarios_and_frequencies.json`: Scenario analysis data

### Model Files
- `dict_model_essentials.pickle`: Model metadata
- `scenario_data.pickle`: Scenario information for inference

## Dependencies

### Core Scientific Packages
- numpy (1.24.4)
- scipy (1.10.1)
- pandas (1.4.3)
- matplotlib (3.5.2)
- scikit-learn (1.1.1)
- networkx (2.8.8)

### Specialized Packages
- pyEDM (1.13.1.0): Empirical Dynamic Modeling
- bnlearn (0.7.13): Bayesian network learning
- pgmpy (0.1.25): Probabilistic graphical models

### Web Interface
- dash (2.9.2): Web application framework
- plotly (5.14.0): Interactive plotting

### Optional LLM Packages
- openai (≥1.0.0): OpenAI API access
- google-generativeai (≥0.3.0): Google AI API access

## Troubleshooting

### Common Issues

**Installation Problems**:
- Ensure Python 3.9 is installed
- Check system dependencies (build-essential, graphviz)
- Try manual package installation

**Memory Issues**:
- Reduce subset length parameter
- Decrease number of cores
- Use smaller datasets for initial testing

**Convergence Problems**:
- Increase subset length
- Try different embedding dimensions
- Check data quality and stationarity

**Web Interface Issues**:
- Clear browser cache
- Check port 8050 availability
- Restart the application

### Getting Help

For technical issues:
1. Check the parameter validation messages
2. Review the debug output in terminal
3. Ensure data format requirements are met
4. Verify sufficient data points for analysis

## Citation

If you use CEcBaN in your research, please cite:

```
Tal, O., Ostrovsky, I., & Gal, G. (2024). A framework for identifying factors controlling cyanobacterium Microcystis flos‐aquae blooms by coupled CCM–ECCM Bayesian networks. Ecology and Evolution, 14(6), e11475.
Hilau, S. and Tal, O. (2025). CEcBaN: A tool for causal analysis of ecological time-series.
```


## Contact

For questions or support, please contact the Tal Lab at the Israel Oceanographic and Limnological Research Institute.
