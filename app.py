import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL
import base64
import io
import pandas as pd
import subprocess
from dash.exceptions import PreventUpdate
import tempfile
import os, re, time
import shutil
import datetime
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import bnlearn as bn
import numpy as np
from itertools import product, combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import seaborn as sns
import copy

try:
    from llm_interactions import get_llm_interactions
    LLM_AVAILABLE = True
    print("LLM interactions module loaded successfully with enhanced features")
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: llm_interactions.py not found. LLM features will be disabled.")

bn_model = None

def visualize_network(results, filepath):
    """Visualize the results of CCM ECCM analysis as a network graph and save as an image."""
    G = nx.DiGraph()
    results = results[results["Score"] > 0]
    results = results[results["is_Valid"] == 2]

    for source, target, weight in results[["x1", "x2", "Score"]].values.tolist():
        G.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', width=3, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title('CCM-ECCM Network')
    plt.savefig(filepath)
    plt.close()

def is_valid_format(s: str) -> bool:
    pattern = r'^\d+[HDWMY]$' 
    return bool(re.match(pattern, s))

def clear_assets_folder():
    assets_folder = 'assets'
    logo_folder = os.path.join(assets_folder, 'logo')
    
    if not os.path.exists(assets_folder):
        os.makedirs(assets_folder)
        return
    
    for filename in os.listdir(assets_folder):
        file_path = os.path.join(assets_folder, filename)
        
        if file_path == logo_folder:
            continue
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def extract_variables_from_filename(filename):
    """Extract variable pair from CCM/ECCM filename - IMPROVED VERSION"""
    try:
        with open('debug_log.txt', 'a') as f:
            f.write(f"\n=== EXTRACT VARIABLES FROM FILENAME ===\n")
            f.write(f"Processing filename: {filename}\n")
        
        # Remove file extension first
        clean_name = filename.replace('.png', '')
        
        # Handle different prefixes
        if clean_name.startswith('ccm_density_'):
            clean_name = clean_name.replace('ccm_density_', '')
            prefix = 'ccm_density'
        elif clean_name.startswith('eccm_'):
            clean_name = clean_name.replace('eccm_', '')
            prefix = 'eccm'
        else:
            with open('debug_log.txt', 'a') as f:
                f.write(f"Unknown prefix in filename: {filename}\n")
            return None, None
        
        with open('debug_log.txt', 'a') as f:
            f.write(f"After prefix removal: {clean_name}, prefix: {prefix}\n")
        
        # Try different separators in order of preference
        separators = ['_vs_', '_to_', '_']
        for sep in separators:
            if sep in clean_name:
                parts = clean_name.split(sep)
                if len(parts) >= 2:
                    x1, x2 = parts[0], parts[1]
                    # Clean up any remaining parts
                    if len(parts) > 2:
                        x2 = '_'.join(parts[1:])
                    
                    with open('debug_log.txt', 'a') as f:
                        f.write(f"Successfully extracted: x1='{x1}', x2='{x2}' using separator '{sep}'\n")
                    return x1, x2
        
        # If no separator found, try to split on underscores and take first two parts
        parts = clean_name.split('_')
        if len(parts) >= 2:
            x1, x2 = parts[0], '_'.join(parts[1:])
            with open('debug_log.txt', 'a') as f:
                f.write(f"Fallback extraction: x1='{x1}', x2='{x2}'\n")
            return x1, x2
        
        with open('debug_log.txt', 'a') as f:
            f.write(f"Failed to extract variables from: {filename}\n")
        return None, None
    except Exception as e:
        with open('debug_log.txt', 'a') as f:
            f.write(f"Exception in extract_variables_from_filename: {e}\n")
        return None, None

def present_all_density_ccm_with_controls(output_folder, current_table_data=None):
    """Present CCM density plots with compact validity controls - FIXED VERSION"""
    prefix = "ccm_density_"
    assets_folder = 'assets/'
    
    image_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(".png")]
    
    with open('debug_log.txt', 'a') as f:
        f.write(f"\n=== CCM PRESENTATION ===\n")
        f.write(f"Found {len(image_files)} CCM files: {image_files}\n")
    
    for img in image_files:
        shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
    
    timestamp = str(int(datetime.datetime.now().timestamp()))
    
    # Create cards with controls for each image
    image_cards = []
    for i, img in enumerate(image_files):
        x1, x2 = extract_variables_from_filename(img)
        
        with open('debug_log.txt', 'a') as f:
            f.write(f"CCM Image {i}: {img} -> x1='{x1}', x2='{x2}'\n")
        
        # Find current validity status from CSV (source of truth)
        current_validity = 2  # default to "Valid"
        csv_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
        if os.path.exists(csv_file) and x1 and x2:
            try:
                df_csv = pd.read_csv(csv_file)
                matching_row = df_csv[(df_csv['x1'] == x1) & (df_csv['x2'] == x2)]
                if not matching_row.empty:
                    current_validity = int(matching_row['is_Valid'].iloc[0])
                    with open('debug_log.txt', 'a') as f:
                        f.write(f"Found CSV row: {x1}->{x2} validity={current_validity}\n")
            except:
                # Fallback to table data if CSV read fails
                if current_table_data and x1 and x2:
                    for row in current_table_data:
                        if row.get('x1') == x1 and row.get('x2') == x2:
                            current_validity = row.get('is_Valid', 2)
                            break
        
        # Create unique ID for this specific image control
        control_id = f'ccm_{x1}_{x2}' if x1 and x2 else f'ccm_image_{i}'
        
        with open('debug_log.txt', 'a') as f:
            f.write(f"CCM Control ID: {control_id}, validity: {current_validity}\n")
        
        card = html.Div(className='result-card image-container', children=[
            # Image
            html.Img(src=f'/assets/{img}?t={timestamp}', 
                    style={'width': '100%', 'height': 'auto', 'display': 'block'}),
            
            # Compact floating control
            html.Div(className='floating-control', children=[
                html.Div(className='control-content', children=[
                    html.Span(f'{x1} → {x2}' if x1 and x2 else f'CCM {i+1}', 
                             className='control-title'),
                    dcc.Dropdown(
                        id={'type': 'ccm-validity', 'index': control_id},
                        options=[
                            {'label': '❌', 'value': 0},
                            {'label': '⚠️', 'value': 1},
                            {'label': '✅', 'value': 2}
                        ],
                        value=current_validity,
                        className='compact-dropdown',
                        clearable=False
                    )
                ])
            ])
        ])
        image_cards.append(card)
    
    return html.Div(className='parameter-card', children=[
        html.H3([html.I(className='fas fa-chart-line', style={'marginRight': '10px'}), 'CCM Analysis Results'], className='card-title'),
        html.P('Hover over images to see validation controls. ❌=Invalid, ⚠️=Under Review, ✅=Valid', 
               style={'color': '#6b7280', 'marginBottom': '20px', 'fontStyle': 'italic', 'fontSize': '0.9rem'}),
        html.Div(className='results-gallery', children=image_cards)
    ])

def present_all_eccm_with_controls(output_folder, current_table_data=None):
    """Present ECCM plots with compact validity controls - FIXED VERSION"""
    prefix = "eccm_"
    assets_folder = 'assets/'
    
    image_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(".png")]
    
    with open('debug_log.txt', 'a') as f:
        f.write(f"\n=== ECCM PRESENTATION ===\n")
        f.write(f"Found {len(image_files)} ECCM files: {image_files}\n")
    
    for img in image_files:
        shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
    
    timestamp = str(int(datetime.datetime.now().timestamp()))
    
    # Create cards with controls for each image
    image_cards = []
    for i, img in enumerate(image_files):
        x1, x2 = extract_variables_from_filename(img)
        
        with open('debug_log.txt', 'a') as f:
            f.write(f"ECCM Image {i}: {img} -> x1='{x1}', x2='{x2}'\n")
        
        # Find current validity status from CSV (source of truth)
        current_validity = 2  # default to "Valid"
        csv_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
        if os.path.exists(csv_file) and x1 and x2:
            try:
                df_csv = pd.read_csv(csv_file)
                matching_row = df_csv[(df_csv['x1'] == x1) & (df_csv['x2'] == x2)]
                if not matching_row.empty:
                    current_validity = int(matching_row['is_Valid'].iloc[0])
                    with open('debug_log.txt', 'a') as f:
                        f.write(f"Found CSV row: {x1}->{x2} validity={current_validity}\n")
            except:
                # Fallback to table data if CSV read fails
                if current_table_data and x1 and x2:
                    for row in current_table_data:
                        if row.get('x1') == x1 and row.get('x2') == x2:
                            current_validity = row.get('is_Valid', 2)
                            break
        
        # Create unique ID for this specific image control
        control_id = f'eccm_{x1}_{x2}' if x1 and x2 else f'eccm_image_{i}'
        
        with open('debug_log.txt', 'a') as f:
            f.write(f"ECCM Control ID: {control_id}, validity: {current_validity}\n")
        
        card = html.Div(className='result-card image-container', children=[
            # Image
            html.Img(src=f'/assets/{img}?t={timestamp}', 
                    style={'width': '100%', 'height': 'auto', 'display': 'block'}),
            
            # Compact floating control
            html.Div(className='floating-control', children=[
                html.Div(className='control-content', children=[
                    html.Span(f'{x1} → {x2}' if x1 and x2 else f'ECCM {i+1}', 
                             className='control-title'),
                    dcc.Dropdown(
                        id={'type': 'eccm-validity', 'index': control_id},
                        options=[
                            {'label': '❌', 'value': 0},
                            {'label': '⚠️', 'value': 1},
                            {'label': '✅', 'value': 2}
                        ],
                        value=current_validity,
                        className='compact-dropdown',
                        clearable=False
                    )
                ])
            ])
        ])
        image_cards.append(card)
    
    return html.Div(className='parameter-card', children=[
        html.H3([html.I(className='fas fa-project-diagram', style={'marginRight': '10px'}), 'ECCM Analysis Results'], className='card-title'),
        html.P('Hover over images to see validation controls. ❌=Invalid, ⚠️=Under Review, ✅=Valid', 
               style={'color': '#6b7280', 'marginBottom': '20px', 'fontStyle': 'italic', 'fontSize': '0.9rem'}),
        html.Div(className='results-gallery', children=image_cards)
    ])

# Keep original functions for compatibility
def present_all_density_ccm(output_folder):
    """Original CCM presentation function"""
    prefix = "ccm_density_"
    assets_folder = 'assets/'
    
    image_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(".png")]
    
    for img in image_files:
        shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
    
    timestamp = str(int(datetime.datetime.now().timestamp()))
    
    return html.Div(className='parameter-card', children=[
        html.H3([html.I(className='fas fa-chart-line', style={'marginRight': '10px'}), 'CCM Analysis Results'], className='card-title'),
        html.Div(className='results-gallery', children=[
            html.Div(className='result-card', children=[
                html.Img(src=f'/assets/{img}?t={timestamp}', style={'width': '100%', 'height': 'auto'})
            ]) for img in image_files
        ])
    ])

def present_all_eccm(output_folder):
    """Original ECCM presentation function"""
    prefix = "eccm_"
    assets_folder = 'assets/'
    
    image_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(".png")]
    
    for img in image_files:
        shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
    
    timestamp = str(int(datetime.datetime.now().timestamp()))
    
    return html.Div(className='parameter-card', children=[
        html.H3([html.I(className='fas fa-project-diagram', style={'marginRight': '10px'}), 'ECCM Analysis Results'], className='card-title'),
        html.Div(className='results-gallery', children=[
            html.Div(className='result-card', children=[
                html.Img(src=f'/assets/{img}?t={timestamp}', style={'width': '100%', 'height': 'auto'})
            ]) for img in image_files
        ])
    ])

app = dash.Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])
app._favicon = ("logo/logo.png")
app.title = 'CEcBaN'
clear_assets_folder()

# Initialize debug log
with open('debug_log.txt', 'w') as f:
    f.write("=== CEcBaN Debug Log ===\n")
    f.write(f"Started at: {datetime.datetime.now()}\n\n")  

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #2d3748;
            }
            
            .main-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                margin: 20px;
                border-radius: 20px;
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
                overflow: hidden;
            }
            
            .header-section {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            
            .header-title {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .header-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                font-weight: 300;
            }
            
            .content-section {
                padding: 40px;
            }
            
            .parameter-card {
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            
            .parameter-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            }
            
            .card-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: #4f46e5;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .upload-zone {
                border: 2px dashed #4f46e5;
                border-radius: 15px;
                background: linear-gradient(135deg, #f8faff 0%, #f1f5ff 100%);
                transition: all 0.3s ease;
            }
            
            .upload-zone:hover {
                border-color: #7c3aed;
                background: linear-gradient(135deg, #f1f5ff 0%, #e5edff 100%);
                transform: scale(1.02);
            }
            
            .modern-button {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
                display: inline-flex;
                align-items: center;
                gap: 10px;
                height: 50px;
            }
            
            .modern-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
            }
            
            .success-button {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            }
            
            .success-button:hover {
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
            }
            
            .secondary-button {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            }
            
            .secondary-button:hover {
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
            }
            
            .input-field {
                width: 100%;
                padding: 12px 16px;
                border: 2px solid #e5e7eb;
                border-radius: 10px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background: white;
                height: 50px;
            }
            
            .input-field:focus {
                outline: none;
                border-color: #4f46e5;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
            }
            
            .progress-section {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin: 30px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                border-left: 5px solid #4f46e5;
            }
            
            .step-indicator {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .step-number {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .step-title {
                font-size: 1.3rem;
                font-weight: 600;
                color: #1f2937;
            }
            
            .results-gallery {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            
            .result-card {
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
                position: relative;
            }
            
            .result-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            }
            
            .result-card img {
                width: 100%;
                height: auto;
                transition: all 0.3s ease;
            }
            
            .download-link {
                transition: all 0.3s ease !important;
            }
            
            .download-link:hover {
                background-color: #4f46e5 !important;
                color: white !important;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            }

            /* PERSISTENT LOADING OVERLAY */
            .persistent-loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .persistent-loading-content {
                background-color: white;
                padding: 50px;
                border-radius: 20px;
                text-align: center;
                min-width: 400px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                animation: fadeIn 0.3s ease-in;
            }

            .persistent-spinner {
                width: 80px;
                height: 80px;
                border: 8px solid #f3f4f6;
                border-top: 8px solid #4f46e5;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 30px auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: scale(0.9); }
                to { opacity: 1; transform: scale(1); }
            }

            .persistent-message {
                font-size: 1.5rem;
                font-weight: 700;
                color: #374151;
                margin: 0 0 15px 0;
            }

            .persistent-details {
                font-size: 1rem;
                color: #6b7280;
                margin: 0;
                line-height: 1.6;
            }

            /* COMPACT IMAGE CONTROL STYLES - IMPROVED */
            .image-container {
                position: relative;
                overflow: visible;
            }

            .floating-control {
                position: absolute;
                top: 8px;
                right: 8px;
                opacity: 0;
                transition: all 0.3s ease;
                z-index: 100;
                pointer-events: none;
            }

            .image-container:hover .floating-control {
                opacity: 1;
                pointer-events: all;
            }

            .control-content {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(8px);
                border-radius: 8px;
                padding: 8px 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
                border: 1px solid rgba(79, 70, 229, 0.3);
                min-width: 120px;
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .control-title {
                font-size: 0.75rem;
                font-weight: 600;
                color: #374151;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 100px;
            }

            .compact-dropdown {
                min-height: 32px !important;
                font-size: 0.8rem !important;
                pointer-events: all !important;
                z-index: 999 !important;
            }

            .compact-dropdown .Select-control {
                min-height: 32px !important;
                height: 32px !important;
                border: 1px solid #d1d5db !important;
                border-radius: 6px !important;
                pointer-events: all !important;
            }

            .compact-dropdown .Select-input {
                height: 30px !important;
                line-height: 30px !important;
            }

            .compact-dropdown .Select-value {
                line-height: 30px !important;
                padding: 0 8px !important;
            }

            .compact-dropdown .Select-arrow-zone {
                width: 25px !important;
                pointer-events: all !important;
            }

            /* Make sure the dropdown menu appears above other content */
            .Select-menu-outer {
                z-index: 1000 !important;
            }
            
            /* Force pointer events for better interaction */
            .floating-control, .floating-control * {
                pointer-events: all !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def write_parameters_to_file(output_folder, step_name, params):
    with open(os.path.join(output_folder, 'parameters.txt'), 'a') as file:
        file.write(f"Step: {step_name}\n")
        for param, value in params.items():
            file.write(f"{param}: {value}\n")
        file.write("\n")

instructions_text = html.Div([
    html.P("CEcBaN is a causal analysis platform implementing CCM-ECCM methodology for identifying causal relationships in time series data.", 
           style={'marginBottom': '10px'}),
    html.P("Upload your CSV data, configure parameters, and discover causal networks with Bayesian modeling capabilities.", 
           style={'marginBottom': '0'})
])

def incorporate_known_interactions(output_folder, known_interactions):
    """Add known interactions to CCM_ECCM_curated.csv"""
    if not known_interactions:
        return "No known interactions to add"
    
    try:
        curated_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
        manual_edits_flag = os.path.join(output_folder, 'MANUAL_EDITS_MADE.flag')
        
        # DON'T OVERWRITE if manual edits have been made
        if os.path.exists(manual_edits_flag):
            return "Manual edits detected - preserving user changes, skipping automatic updates"
        
        if os.path.exists(curated_file):
            df_existing = pd.read_csv(curated_file)
        else:
            df_existing = pd.DataFrame(columns=['x1', 'x2', 'Score', 'is_Valid', 'timeToEffect'])
        
        known_data = []
        updated_existing = []
        
        for interaction in known_interactions:
            x1, x2, score, lag = interaction
            
            existing = df_existing[(df_existing['x1'] == x1) & (df_existing['x2'] == x2)]
            if existing.empty:
                known_data.append({
                    'x1': x1,
                    'x2': x2,
                    'Score': float(score),
                    'is_Valid': 2,  
                    'timeToEffect': int(lag)
                })
            else:
                df_existing.loc[(df_existing['x1'] == x1) & (df_existing['x2'] == x2), 'is_Valid'] = 2
                df_existing.loc[(df_existing['x1'] == x1) & (df_existing['x2'] == x2), 'Score'] = float(score)
                df_existing.loc[(df_existing['x1'] == x1) & (df_existing['x2'] == x2), 'timeToEffect'] = int(lag)
                updated_existing.append(f"{x1} → {x2} (Score: {score}, Lag: {lag})")
        
        if known_data:
            df_known = pd.DataFrame(known_data)
            df_combined = pd.concat([df_existing, df_known], ignore_index=True)
        else:
            df_combined = df_existing
        
        initial_length = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['x1', 'x2'], keep='first')
        final_length = len(df_combined)
        
        if 'Score' in df_combined.columns:
            df_combined['Score'] = pd.to_numeric(df_combined['Score'], errors='coerce').fillna(0.8)
        if 'is_Valid' in df_combined.columns:
            df_combined['is_Valid'] = pd.to_numeric(df_combined['is_Valid'], errors='coerce').fillna(2)
        if 'timeToEffect' in df_combined.columns:
            df_combined['timeToEffect'] = pd.to_numeric(df_combined['timeToEffect'], errors='coerce').fillna(0)
        
        df_combined.to_csv(curated_file, index=False)
        
        status_parts = []
        if known_data:
            new_interactions = [f"{item['x1']} → {item['x2']} (Score: {item['Score']}, Lag: {item['timeToEffect']})" for item in known_data]
            status_parts.append(f"Added {len(new_interactions)} new known interactions: {', '.join(new_interactions)}")
        
        if updated_existing:
            status_parts.append(f"Updated {len(updated_existing)} existing interactions: {', '.join(updated_existing)}")
        
        if initial_length != final_length:
            status_parts.append(f"Removed {initial_length - final_length} duplicates")
        
        valid_count = len(df_combined[df_combined['is_Valid'] == 2])
        status_parts.append(f"File now contains {valid_count} valid interactions total")
        
        return "; ".join(status_parts) if status_parts else "Known interactions processed"
        
    except Exception as e:
        return f"Error incorporating known interactions: {str(e)}"

app.layout = html.Div([
    html.Div(id='persistent-loading', style={'display': 'none'}, children=[
        html.Div(className='persistent-loading-overlay', children=[
            html.Div(className='persistent-loading-content', children=[
                html.Div(className='persistent-spinner'),
                html.H4(id='persistent-message', className='persistent-message', children='Processing...'),
                html.P(id='persistent-details', className='persistent-details', children='This may take several minutes to hours...')
            ])
        ])
    ]),

    dcc.Store(id='stored-output-folder', storage_type='session'),
    dcc.Store(id='image-control-updates', storage_type='memory'),  # Track manual updates
    
    html.Div(className='main-container', children=[  
        html.Div(className='header-section', children=[
            html.Div([
                html.Img(src='/assets/logo/logo.png', style={'width': '80px', 'height': 'auto', 'marginRight': '20px'}),
                html.Div([
                    html.H1('CEcBaN', className='header-title'),
                    html.P('CCM ECCM Bayesian Network Analysis Platform', className='header-subtitle'),
                ], style={'flex': '1'}),
                html.Img(src='/assets/logo/logo_iolr.png', style={'width': '80px', 'height': 'auto', 'marginLeft': '20px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
            html.P('Tal Lab - Advanced Causal Analysis Tool', style={'marginTop': '20px', 'fontSize': '1.1rem', 'opacity': '0.8'})
        ]),
        
        html.Div(className='content-section', children=[
            
            html.Div(className='parameter-card', children=[
                html.H3([html.I(className='fas fa-info-circle', style={'marginRight': '10px'}), 'About CEcBaN'], className='card-title'),
                instructions_text,
                
                html.Div([
                    html.H5([html.I(className='fas fa-download', style={'marginRight': '8px'}), 'Resources'], 
                            style={'color': '#4f46e5', 'marginBottom': '15px', 'marginTop': '20px'}),
                    
                    html.Div([
                        html.A([
                            html.I(className='fas fa-file-pdf', style={'marginRight': '8px', 'color': '#dc2626'}),
                            'Download Tal et al. 2024 Paper'
                        ], 
                        id='download-tal-paper-link',
                        href='#',
                        className='download-link',
                        style={
                            'color': '#4f46e5', 
                            'textDecoration': 'none', 
                            'fontSize': '0.95rem',
                            'fontWeight': '500',
                            'padding': '10px 15px',
                            'border': '1px solid #4f46e5',
                            'borderRadius': '8px',
                            'display': 'inline-block',
                            'transition': 'all 0.3s ease',
                            'marginRight': '15px',
                            'marginBottom': '10px'
                        }),
                        
                        html.A([
                            html.I(className='fas fa-file-alt', style={'marginRight': '8px', 'color': '#059669'}),
                            'Download Instructions Guide'
                        ], 
                        id='download-instructions-link',
                        href='#',
                        className='download-link',
                        style={
                            'color': '#4f46e5', 
                            'textDecoration': 'none', 
                            'fontSize': '0.95rem',
                            'fontWeight': '500',
                            'padding': '10px 15px',
                            'border': '1px solid #4f46e5',
                            'borderRadius': '8px',
                            'display': 'inline-block',
                            'transition': 'all 0.3s ease',
                            'marginBottom': '10px'
                        })
                    ], style={'display': 'block'}),
                    
                    dcc.Download(id='download-tal-paper'),
                    dcc.Download(id='download-instructions')
                ])
            ]),
            
            html.Div(className='parameter-card', children=[
                html.H3([html.I(className='fas fa-upload', style={'marginRight': '10px'}), 'Data Upload'], className='card-title'),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.I(className='fas fa-cloud-upload-alt', style={'fontSize': '3rem', 'color': '#4f46e5', 'marginBottom': '15px'}),
                        html.Div(['Drag and Drop or ', html.A('Select Files', style={'color': '#4f46e5', 'fontWeight': '600'})]),
                        html.P('Supported format: CSV files', style={'fontSize': '0.9rem', 'color': '#6b7280', 'marginTop': '10px'})
                    ], style={'textAlign': 'center', 'padding': '40px'}),
                    className='upload-zone',
                    style={'width': '100%', 'minHeight': '150px', 'lineHeight': '60px'},
                    multiple=False
                ),
                html.Div(id='output-data-upload', style={'marginTop': '20px'})
            ]),
            
            dcc.Store(id='stored-filename'),  
            dcc.Store(id='stored-columns'), 
            dcc.Store(id='categorization-filepath'),  
            dcc.Store(id='first-step-completed', data=False),
            dcc.Store(id='inference-model-data'),
            dcc.Store(id='known-interactions', data=[]),
            dcc.Store(id='llm-interactions', data=[]),
            dcc.Store(id='removed-llm-interactions', data=[]),
            dcc.Store(id='table-sync-status', data={}),
            
            html.Div(className='parameter-card', children=[
                html.H3([html.I(className='fas fa-cog', style={'marginRight': '10px'}), 'Analysis Configuration'], className='card-title'),
                
                html.Div([
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Target Column'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(id='target-column', className='input-field', style={'marginBottom': '20px'}),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Confounders'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Checklist(id='confounder-columns', style={'marginBottom': '20px'}),
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Subset Length'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='subset-length', type='number', value=60, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Jump N'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='jump-n', type='number', value=30, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Z-score Threshold'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='z-score-threshold', type='number', value=3, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Resample Freq'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='resample-freq', type='text', value='5D', className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Embedding Dimension'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='embedding-dim', type='number', value=2, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Lag'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='lag', type='number', value=1, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'ECCM Window'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='eccm-window-size', type='number', value=50, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'CPU Cores'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='number-of-cores', type='number', value=1, className='input-field'),
                    ], style={'width': '23%', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'CCM Training Proportion'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='ccm-training-proportion', type='number', value=0.75, min=0, max=1, step=0.01, className='input-field'),
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Max MI Shift'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(id='max-mutual-info-shift', type='number', value=20, className='input-field'),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ]),
                
                html.Div([
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'Convergence Method'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='check-convergence',
                            options=[
                                {'label': 'Results Mean', 'value': 'means'},
                                {'label': 'Results Density', 'value': 'density'}
                            ],
                            value='density',
                            className='input-field'
                        )
                    ], style={'width': '48%', 'marginTop': '20px', 'marginRight': '4%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label([html.I(className='fas fa-circle', style={'marginRight': '8px'}), 'ECCM Lag Strategy'], style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='prefer-zero-lag',
                            options=[
                                {'label': 'Prefer Immediate Effects', 'value': 'true'},
                                {'label': 'Use Strongest Effects', 'value': 'false'}
                            ],
                            value='true',
                            className='input-field'
                        )
                    ], style={'width': '48%', 'marginTop': '20px', 'display': 'inline-block'})
                ])
            ]),
            
            html.Div(id='known-interactions-section', className='parameter-card', style={'display': 'none'}, children=[
                html.H3([html.I(className='fas fa-link', style={'marginRight': '10px'}), 'Known Interactions'], className='card-title'),
                html.P("Specify variable pairs that you know affect each other, or use AI to discover interactions from scientific literature.", 
                       style={'marginBottom': '20px', 'color': '#6b7280'}),
                
                html.Div(className='parameter-card', style={'backgroundColor': '#f0f9ff', 'border': '1px solid #0ea5e9', 'marginBottom': '25px'}, children=[
                    html.H4([html.I(className='fas fa-robot', style={'marginRight': '10px'}), 'AI Literature Discovery'], 
                            style={'color': '#0ea5e9', 'marginBottom': '15px'}),
                    
                    html.Div([
                        html.Div([
                            html.Label('LLM Provider:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='llm-provider',
                                options=[
                                    {'label': 'OpenAI GPT-4', 'value': 'openai'},
                                    {'label': 'Google Gemini', 'value': 'google'}
                                ],
                                value='openai',
                                className='input-field',
                                style={'minHeight': '50px', 'height': '50px'}
                            )
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.Label('API Key:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Input(
                                id='llm-api-key',
                                type='password',
                                placeholder='Enter your API key...',
                                className='input-field',
                                style={'height': '50px'}
                            )
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Button([
                            html.I(className='fas fa-brain', style={'marginRight': '10px'}),
                            'Ask AI for Literature-Based Interactions'
                        ], id='ask-llm-button', n_clicks=0, className='modern-button', 
                        style={'marginRight': '15px'}),
                        
                        dcc.Loading(
                            id="loading-llm",
                            type="graph",
                            color="#0ea5e9",
                            children=html.Div(id="llm-status", style={'display': 'inline-block', 'marginLeft': '15px'})
                        )
                    ], style={'textAlign': 'center'}),
                    
                    html.Div(id='llm-results-display', style={'marginTop': '20px'})
                ]),
                
                html.Div([
                    html.H4([html.I(className='fas fa-edit', style={'marginRight': '10px'}), 'Manual Entry'], 
                            style={'color': '#4f46e5', 'marginBottom': '15px'}),
                    
                    html.Div(id='known-interactions-container', children=[
                        html.Div(id='known-interaction-0', children=[
                            html.Div([
                                html.Div([
                                    html.Label('Source Variable', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                                    dcc.Dropdown(id={'type': 'known-x1', 'index': 0}, className='input-field', style={'height': '46px'}),
                                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                                
                                html.Div([
                                    html.Label('Target Variable', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                                    dcc.Dropdown(id={'type': 'known-x2', 'index': 0}, className='input-field', style={'height': '46px'}),
                                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                                
                                html.Div([
                                    html.Label('Score (0-1)', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                                    dcc.Input(id={'type': 'known-score', 'index': 0}, type='number', min=0, max=1, step=0.1, value=0.8, className='input-field', style={'height': '46px'}),
                                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                                
                                html.Div([
                                    html.Label('Time Lag', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                                    dcc.Input(id={'type': 'known-lag', 'index': 0}, type='number', min=0, step=1, value=0, className='input-field', style={'height': '46px'}),
                                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                                
                                html.Div([
                                    html.Label('', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                                    html.Button([html.I(className='fas fa-trash')], 
                                               id={'type': 'remove-known', 'index': 0}, 
                                               className='modern-button', 
                                               style={'backgroundColor': '#ef4444', 'padding': '10px 15px', 'height': '46px', 'width': '100%'}),
                                ], style={'width': '12%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'top'})
                            ], style={'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '10px'})
                        ])
                    ]),
                    
                    html.Div(style={'textAlign': 'center', 'marginTop': '20px'}, children=[
                        html.Button([
                            html.I(className='fas fa-plus', style={'marginRight': '8px'}),
                            'Add Manual Interaction'
                        ], id='add-known-interaction', n_clicks=0, className='modern-button secondary-button')
                    ])
                ])
            ]),
            
            html.Div(style={'textAlign': 'center', 'margin': '40px 0'}, children=[
                html.Button([
                    html.I(className='fas fa-play', style={'marginRight': '10px'}),
                    'Start CCM ECCM Analysis'
                ], id='run-pipeline', n_clicks=0, className='modern-button', style={'fontSize': '1.1rem', 'padding': '18px 40px'})
            ]),
            
            html.Div([
                dcc.Loading(
                    id="loading-pipeline",
                    type="graph",
                    color="#4f46e5",
                    children=html.Div(id="loading-output-pipeline"),
                    style={'minHeight': '100px'}
                )
            ]),
            
            html.Div(id='post-pipeline-step', style={'display': 'none'}),  
            html.Div(id='pipeline-output'),
            
            html.Div(id='first-step-results'),
            
            html.Div(id='second-step', children=[
                html.Div(className='progress-section', children=[
                    html.Div(className='step-indicator', children=[
                        html.Div('2', className='step-number'),
                        html.H2('Network Refinement', className='step-title')
                    ]),
                    
                    html.Div(className='parameter-card', children=[
                        html.H4([html.I(className='fas fa-edit', style={'marginRight': '10px'}), 'Edit Interactions Network'], className='card-title'),
                        html.P("Review and refine the network structure. Use the dropdown to validate connections.", 
                               style={'marginBottom': '20px', 'color': '#6b7280'}),
                        
                        dash_table.DataTable(
                            id='table-editing',
                            columns=[],
                            data=[],
                            dropdown={},
                            editable=True,
                            row_deletable=True,
                            page_action='none',
                            style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                            style_cell={
                                'textAlign': 'left', 
                                'minWidth': '120px', 
                                'padding': '12px',
                                'fontFamily': 'Inter',
                                'fontSize': '0.9rem',
                                'border': '1px solid #e5e7eb'
                            },
                            style_header={
                                'backgroundColor': '#f8fafc',
                                'fontWeight': '600',
                                'color': '#374151',
                                'border': '1px solid #d1d5db'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'is_Valid', 'filter_query': '{is_Valid} = 2'},
                                    'backgroundColor': '#d1fae5',
                                    'color': '#065f46',
                                },
                                {
                                    'if': {'column_id': 'is_Valid', 'filter_query': '{is_Valid} = 0'},
                                    'backgroundColor': '#fee2e2',
                                    'color': '#991b1b',
                                },
                                {
                                    'if': {'column_id': 'is_Valid', 'filter_query': '{is_Valid} = 1'},
                                    'backgroundColor': '#fed7aa',
                                    'color': '#9a3412',
                                }
                            ]
                        ),
                        
                        html.Div([
                            html.Button([
                                html.I(className='fas fa-plus', style={'marginRight': '8px'}),
                                'Add Connection'
                            ], id='add-row-button', n_clicks=0, className='modern-button secondary-button', style={'marginRight': '15px'}),
                            
                            html.Button([
                                html.I(className='fas fa-save', style={'marginRight': '8px'}),
                                'Save Changes'
                            ], id='save-edits', n_clicks=0, className='modern-button success-button', style={'display': 'none'}),
                        ], style={'marginTop': '20px'}),
                        
                        html.Div(id='save-confirmation', style={'marginTop': '20px'}),
                    ]),
                    
                    html.Div(className='parameter-card', children=[
                        html.H4([html.I(className='fas fa-random', style={'marginRight': '10px'}), 'Surrogate Analysis'], className='card-title'),
                        
                        html.Div([
                            html.Div([
                                html.Label('Number of Surrogates for x1:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(id='num-surrogates-x1', type='number', value=33, className='input-field'),
                            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                            
                            html.Div([
                                html.Label('Number of Surrogates for x2:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(id='num-surrogates-x2', type='number', value=33, className='input-field'),
                            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                            
                            html.Div([
                                html.Label('Significance Quantile:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(id='sig-quant', type='number', value=0.95, className='input-field'),
                            ], style={'width': '32%', 'display': 'inline-block'}),
                        ]),
                        
                        html.Div([
                            dcc.Loading(
                                id="loading-pipeline2",
                                type="graph",
                                color="#4f46e5",
                                children=html.Div(id="loading-output-pipeline2"),
                                style={'minHeight': '80px'}
                            )
                        ], style={'marginTop': '20px'}),
                        
                        html.Div(style={'textAlign': 'center', 'marginTop': '30px'}, children=[
                            html.Button([
                                html.I(className='fas fa-chart-bar', style={'marginRight': '10px'}),
                                'Run Surrogate Analysis'
                            ], id='run-second-step', n_clicks=0, className='modern-button')
                        ])
                    ])
                ]),
                
                html.Div(id='second-step-results')
            ], style={'display': 'none'}),
            
            html.Div(id='third-step', children=[
                html.Div(className='progress-section', children=[
                    html.Div(className='step-indicator', children=[
                        html.Div('4', className='step-number'),
                        html.H2('Bayesian Network Modeling', className='step-title')
                    ]),
                    
                    html.Div(className='parameter-card', children=[
                        html.H4([html.I(className='fas fa-network-wired', style={'marginRight': '10px'}), 'Model Configuration'], className='card-title'),
                        
                        html.Div([
                            html.Label([html.I(className='fas fa-tags', style={'marginRight': '8px'}), 'Categorization Method'], style={'fontWeight': '600', 'marginBottom': '15px', 'display': 'block'}),
                            dcc.RadioItems(
                                id='categorization-mode',
                                options=[
                                    {'label': ' Automatic', 'value': 'auto'},
                                    {'label': ' Upload Categories File', 'value': 'upload'}
                                ],
                                value='auto',
                                labelStyle={'display': 'block', 'marginBottom': '10px', 'fontWeight': '500'}
                            ),
                            
                            html.Div([
                                html.A([
                                    html.I(className='fas fa-download', style={'marginRight': '8px'}),
                                    'Download Categorization Format Example'
                                ], 
                                id='download-categories-format-link',
                                href='#',
                                className='download-link',
                                style={
                                    'color': '#4f46e5', 
                                    'textDecoration': 'none', 
                                    'fontSize': '0.9rem',
                                    'fontWeight': '500',
                                    'padding': '8px 12px',
                                    'border': '1px solid #4f46e5',
                                    'borderRadius': '6px',
                                    'display': 'inline-block',
                                    'transition': 'all 0.3s ease'
                                }),
                                dcc.Download(id='download-categories-format')
                            ], style={'marginTop': '10px'})
                        ], style={'marginBottom': '25px'}),
                        
                        html.Div(id='upload-categorization-file', children=[
                            html.Label('Upload Categories File:', style={'fontWeight': '600', 'marginBottom': '10px', 'display': 'block'}),
                            dcc.Upload(
                                id='upload-categories-file',
                                children=html.Div([
                                    html.I(className='fas fa-file-csv', style={'fontSize': '2rem', 'color': '#4f46e5', 'marginBottom': '10px'}),
                                    html.Div(['Drop categories file here']),
                                ], style={'textAlign': 'center', 'padding': '30px'}),
                                className='upload-zone',
                                style={'minHeight': '120px'},
                                multiple=False
                            ),
                            html.Div(id='output-categories-upload', style={'marginTop': '15px'})
                        ], style={'display': 'none'}),

                        html.Div([
                            html.Div([
                                html.Label('Training Fraction:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(id='bn-training-fraction', type='number', value=0.75, min=0, max=1, step=0.01, className='input-field'),
                            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                            
                            html.Div([
                                html.Label('Probability Cutoff:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(id='probability-cutoff', type='number', value=0.5, className='input-field'),
                            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                            
                            html.Div([
                                html.Label('Bidirectional Rule:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='bidirectional-interaction',
                                    options=[
                                        {'label': 'Higher CCM Score', 'value': 'higher'},
                                        {'label': 'Earlier Effect', 'value': 'earlier'}
                                    ],
                                    value='higher',
                                    className='input-field',
                                    style={'minHeight': '46px'}
                                )
                            ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        ]),
                        
                        html.Div([
                            dcc.Loading(
                                id="loading-pipeline3",
                                type="graph",
                                color="#4f46e5",
                                children=html.Div(id="loading-output-pipeline3"),
                                style={'minHeight': '80px'}
                            )
                        ], style={'marginTop': '30px'}),
                        
                        html.Div(style={'textAlign': 'center', 'marginTop': '30px'}, children=[
                            html.Button([
                                html.I(className='fas fa-brain', style={'marginRight': '10px'}),
                                'Build Bayesian Network'
                            ], id='run-final-step', n_clicks=0, className='modern-button')
                        ])
                    ])
                ]),
                
                html.Div(id='third-step-results')
            ], style={'display': 'none'}),
            
            html.Div(id='fourth-step', children=[
                html.Div(className='progress-section', children=[
                    html.Div(className='step-indicator', children=[
                        html.Div('6', className='step-number'),
                        html.H2('Model Inference & Prediction', className='step-title')
                    ]),
                    
                    html.Div(id='model-stats-section', children=[], style={'display': 'none'}),
                    
                    html.Div(id='inference-section', children=[], style={'display': 'none'})
                ])
            ], style={'display': 'none'}),
        ])
    ])
])

# FIXED CALLBACKS FOR IMAGE CONTROLS - PREVENT CIRCULAR TRIGGERS
@app.callback(
    Output('image-control-updates', 'data'),
    [Input({'type': 'ccm-validity', 'index': ALL}, 'value'),
     Input({'type': 'eccm-validity', 'index': ALL}, 'value')],
    [State({'type': 'ccm-validity', 'index': ALL}, 'id'),
     State({'type': 'eccm-validity', 'index': ALL}, 'id'),
     State('image-control-updates', 'data'),
     State('stored-output-folder', 'data')],
    prevent_initial_call=True
)
def track_image_control_changes(ccm_values, eccm_values, ccm_ids, eccm_ids, current_updates, output_folder):
    """Track manual changes to image controls to prevent circular updates - FIXED VERSION"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Debug file logging with more detail
    with open('debug_log.txt', 'a') as f:
        f.write(f"\n=== TRACK IMAGE CONTROL CHANGES ===\n")
        f.write(f"Time: {datetime.datetime.now()}\n")
        f.write(f"Triggered by: {[t['prop_id'] for t in ctx.triggered]}\n")
        f.write(f"Trigger values: {[t['value'] for t in ctx.triggered]}\n")
    
    updates = current_updates or {}
    
    # Only process the specific control that actually triggered the change
    triggered_prop_ids = [t['prop_id'] for t in ctx.triggered]
    
    # Track CCM changes - only for triggered controls
    for value, id_dict in zip(ccm_values, ccm_ids):
        # FIXED: Use double quotes to match actual trigger format
        control_prop_id = f'{{"index":"{id_dict["index"]}","type":"ccm-validity"}}.value'
        
        if control_prop_id in triggered_prop_ids and value is not None:
            control_id = id_dict['index']
            with open('debug_log.txt', 'a') as f:
                f.write(f"TRIGGERED CCM Control ID: {control_id}, Value: {value} (type: {type(value)})\n")
                f.write(f"Matched prop ID: {control_prop_id}\n")
                
            # Extract variable names from control ID - IMPROVED VERSION
            if control_id.startswith('ccm_'):
                var_part = control_id[4:]  # Remove 'ccm_' prefix
                if '_' in var_part:
                    # Try to split into exactly 2 parts
                    underscore_pos = var_part.find('_')
                    if underscore_pos != -1:
                        x1 = var_part[:underscore_pos]
                        x2 = var_part[underscore_pos+1:]
                        key = f"{x1}_{x2}"
                        
                        # Add unique timestamp to ensure this is a fresh update
                        import time
                        updates[key] = {
                            'value': value, 
                            'source': 'image', 
                            'timestamp': datetime.datetime.now().isoformat(),
                            'trigger_id': control_prop_id,
                            'unique_id': time.time()  # Force uniqueness
                        }
                        with open('debug_log.txt', 'a') as f:
                            f.write(f"TRIGGERED CCM Update stored: {key} = {value}\n")
    
    # Track ECCM changes - only for triggered controls
    for value, id_dict in zip(eccm_values, eccm_ids):
        # FIXED: Use double quotes to match actual trigger format
        control_prop_id = f'{{"index":"{id_dict["index"]}","type":"eccm-validity"}}.value'
        
        if control_prop_id in triggered_prop_ids and value is not None:
            control_id = id_dict['index']
            with open('debug_log.txt', 'a') as f:
                f.write(f"TRIGGERED ECCM Control ID: {control_id}, Value: {value} (type: {type(value)})\n")
                f.write(f"Matched prop ID: {control_prop_id}\n")
                
            # Extract variable names from control ID - IMPROVED VERSION
            if control_id.startswith('eccm_'):
                var_part = control_id[5:]  # Remove 'eccm_' prefix
                if '_' in var_part:
                    # Try to split into exactly 2 parts
                    underscore_pos = var_part.find('_')
                    if underscore_pos != -1:
                        x1 = var_part[:underscore_pos]
                        x2 = var_part[underscore_pos+1:]
                        key = f"{x1}_{x2}"
                        
                        # Add unique timestamp to ensure this is a fresh update
                        import time
                        updates[key] = {
                            'value': value, 
                            'source': 'image', 
                            'timestamp': datetime.datetime.now().isoformat(),
                            'trigger_id': control_prop_id,
                            'unique_id': time.time()  # Force uniqueness
                        }
                        with open('debug_log.txt', 'a') as f:
                            f.write(f"TRIGGERED ECCM Update stored: {key} = {value}\n")
    
    with open('debug_log.txt', 'a') as f:
        f.write(f"Final updates: {updates}\n")
        f.write(f"Number of triggered updates: {len([k for k, v in updates.items() if v.get('unique_id')])}\n")
    
    # Write changes directly to CSV (single source of truth)
    if updates and output_folder:
        try:
            csv_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                # Apply updates to CSV data
                for var_pair, update_info in updates.items():
                    if update_info.get('source') == 'image' and update_info.get('unique_id'):
                        parts = var_pair.split('_')
                        if len(parts) >= 2:
                            x1, x2 = parts[0], '_'.join(parts[1:])
                            value = update_info['value']
                            
                            # Find and update the corresponding row in CSV
                            for i, row in df.iterrows():
                                if str(row.get('x1', '')).strip() == str(x1).strip() and str(row.get('x2', '')).strip() == str(x2).strip():
                                    df.at[i, 'is_Valid'] = value
                                    with open('debug_log.txt', 'a') as f:
                                        f.write(f"CSV UPDATED: {x1} → {x2} = {value}\n")
                                    break
                
                # Save updated CSV
                df.to_csv(csv_file, index=False)
                with open('debug_log.txt', 'a') as f:
                    f.write(f"CSV SAVED with image control changes\n")
                    
        except Exception as e:
            with open('debug_log.txt', 'a') as f:
                f.write(f"CSV UPDATE ERROR: {str(e)}\n")
    
    return updates

# Removed update_table_from_image_controls callback - table reads directly from CSV

# Direct table CSV sync
@app.callback(
    Output('table-sync-status', 'data'),
    [Input('table-editing', 'data')],
    [State('stored-output-folder', 'data')],
    prevent_initial_call=True
)
def sync_table_to_csv(table_data, output_folder):
    if not table_data or not output_folder:
        raise PreventUpdate
    
    try:
        curated_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
        df_updated = pd.DataFrame(table_data)
        df_updated.to_csv(curated_file, index=False)
        
            
        return {'synced': True}
    except:
        return {'synced': False}

# Clear image control store when table is edited directly to force sync from CSV
@app.callback(
    Output('image-control-updates', 'data', allow_duplicate=True),
    [Input('table-sync-status', 'data')],
    prevent_initial_call=True
)
def clear_image_store_on_table_edit(sync_status):
    """Clear image control store when table is edited to force fresh sync from CSV"""
    if sync_status and sync_status.get('synced'):
        with open('debug_log.txt', 'a') as f:
            f.write(f"🧹 CLEARING image control store due to direct table edit\n")
        return {}  # Clear the store
    raise PreventUpdate

# Refresh table from CSV when image controls change
@app.callback(
    Output('table-editing', 'data', allow_duplicate=True),
    [Input('image-control-updates', 'data')],
    [State('stored-output-folder', 'data')],
    prevent_initial_call=True
)
def refresh_table_from_csv(image_updates, output_folder):
    """Refresh table data from CSV when image controls change"""
    if not image_updates or not output_folder:
        raise PreventUpdate
    
    # Only refresh if there are actual changes with unique_id (real user changes)
    has_real_changes = any(update.get('unique_id') for update in image_updates.values())
    if not has_real_changes:
        raise PreventUpdate
    
    try:
        csv_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            with open('debug_log.txt', 'a') as f:
                f.write(f"TABLE REFRESHED from CSV due to image control changes\n")
            return df.to_dict('records')
    except Exception as e:
        with open('debug_log.txt', 'a') as f:
            f.write(f"TABLE REFRESH ERROR: {str(e)}\n")
    
    raise PreventUpdate

@app.callback(
    Output('persistent-loading', 'style', allow_duplicate=True),
    [Input('loading-output-pipeline', 'children')],
    prevent_initial_call=True
)
def hide_loading_after_stage1(loading_output):
    return {'display': 'none'}

@app.callback(
    Output('persistent-loading', 'style', allow_duplicate=True), 
    [Input('loading-output-pipeline2', 'children')],
    prevent_initial_call=True
)
def hide_loading_after_stage2(loading_output):
    return {'display': 'none'}

@app.callback(
    Output('persistent-loading', 'style', allow_duplicate=True),
    [Input('loading-output-pipeline3', 'children')], 
    prevent_initial_call=True
)
def hide_loading_after_stage3(loading_output):
    return {'display': 'none'}

@app.callback(
    [Output('llm-interactions', 'data'),
     Output('llm-status', 'children'),
     Output('llm-results-display', 'children')],
    [Input('ask-llm-button', 'n_clicks')],
    [State('stored-columns', 'data'),
     State('llm-provider', 'value'),
     State('llm-api-key', 'value'),
     State('target-column', 'value'),
     State('resample-freq', 'value')]
)
def query_llm_for_interactions(n_clicks, columns, provider, api_key, target_column, resample_freq):
    if n_clicks == 0 or not columns or not LLM_AVAILABLE:
        if not LLM_AVAILABLE:
            return [], html.Div("LLM module not available", style={'color': 'red'}), []
        return [], '', []
    
    if not api_key or not provider:
        return [], html.Div("Please provide API key and select provider", style={'color': 'red'}), []
    
    if not target_column:
        return [], html.Div("Please select a target variable first", style={'color': 'red'}), []
    
    try:
        # Filter out date columns for LLM query
        variables = [col for col in columns if col.lower() not in ['date', 'index']]
        
        if len(variables) < 2:
            return [], html.Div("Need at least 2 variables for interaction discovery", style={'color': 'orange'}), []
        
        llm_interactions, status_msg = get_llm_interactions(
            variables, 
            provider=provider, 
            api_key=api_key,
            target_variable=target_column,
            time_resolution=resample_freq,
            debug=True 
        )
        
        
        if llm_interactions:
            interaction_cards = []
            for i, interaction in enumerate(llm_interactions):
                source, target, score, lag, justification = interaction
                
                if '\n\nReference:' in justification:
                    main_text, reference = justification.split('\n\nReference:', 1)
                    reference = reference.strip()
                else:
                    main_text = justification
                    reference = ""
                
                card = html.Div([
                    html.Div([
                        html.Div([
                            html.Strong(f"{source} → {target}", style={'fontSize': '1.1rem', 'color': '#374151'}),
                            html.Span(f" (Score: {score:.2f}, Lag: {lag})", 
                                    style={'color': '#6b7280', 'marginLeft': '10px', 'fontSize': '0.9rem'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
                        
                        html.P(main_text, style={
                            'fontSize': '0.85rem', 
                            'color': '#4b5563', 
                            'margin': '0 0 8px 0',
                            'fontStyle': 'italic',
                            'lineHeight': '1.4'
                        }),
                        
                        html.Div([
                            html.Strong("Reference: ", style={'fontSize': '0.8rem', 'color': '#374151'}),
                            html.Span(reference, style={
                                'fontSize': '0.8rem', 
                                'color': '#6b7280',
                                'lineHeight': '1.3'
                            })
                        ], style={'marginTop': '8px'}) if reference else None
                        
                    ], style={'flex': '1', 'marginRight': '15px'}),
                    
                    html.Button([
                        html.I(className='fas fa-trash', style={'fontSize': '0.9rem'})
                    ], 
                    id={'type': 'remove-llm-interaction', 'index': i},
                    className='modern-button',
                    style={
                        'backgroundColor': '#ef4444',
                        'padding': '8px 12px',
                        'minWidth': '40px',
                        'height': '40px'
                    },
                    title=f"Remove {source} → {target} interaction")
                    
                ], style={
                    'display': 'flex', 
                    'alignItems': 'flex-start', 
                    'padding': '15px', 
                    'backgroundColor': '#f0fdf4', 
                    'margin': '8px 0', 
                    'borderRadius': '8px',
                    'border': '1px solid #bbf7d0'
                })
                
                interaction_cards.append(card)
            
            results_display = html.Div([
                html.H5([
                    html.I(className='fas fa-check-circle', style={'marginRight': '10px', 'color': '#10b981'}), 
                    f'Found {len(llm_interactions)} literature-based interactions for {target_column} (resolution: {resample_freq})'
                ], style={'color': '#10b981', 'marginBottom': '15px'}),
                
                html.Div(interaction_cards, style={'maxHeight': '400px', 'overflowY': 'auto'}),
                
                html.P("Click the trash icon to remove unwanted interactions. Remaining interactions will be automatically added to your analysis.", 
                       style={'marginTop': '15px', 'color': '#059669', 'fontStyle': 'italic', 'fontSize': '0.9rem'})
            ])
            
            status_display = html.Div(status_msg, style={'color': '#10b981', 'fontWeight': '600'})
        else:
            results_display = html.Div([
                html.H5([html.I(className='fas fa-info-circle', style={'marginRight': '10px', 'color': '#f59e0b'}), 
                        'No interactions found'], style={'color': '#f59e0b'}),
                html.P(f"The LLM did not identify any high-confidence interactions for target variable '{target_column}' at {resample_freq} resolution.", 
                       style={'color': '#6b7280'})
            ])
            status_display = html.Div(status_msg, style={'color': '#f59e0b'})
        
        return llm_interactions, status_display, results_display
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        error_display = html.Div([
            html.H5([html.I(className='fas fa-exclamation-triangle', style={'marginRight': '10px', 'color': '#ef4444'}), 
                    'Query Failed'], style={'color': '#ef4444'}),
            html.P(error_msg, style={'color': '#6b7280'})
        ])
        return [], html.Div(error_msg, style={'color': '#ef4444'}), error_display

@app.callback(
    Output('removed-llm-interactions', 'data'),
    [Input({'type': 'remove-llm-interaction', 'index': ALL}, 'n_clicks')],
    [State('removed-llm-interactions', 'data'),
     State('llm-interactions', 'data')]
)
def remove_llm_interaction(n_clicks_list, removed_interactions, llm_interactions):
    if not any(n_clicks_list) or not llm_interactions:
        return removed_interactions or []
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return removed_interactions or []
    
    button_id = ctx.triggered[0]['prop_id']
    if 'remove-llm-interaction' in button_id:
        import json
        button_info = json.loads(button_id.split('.')[0])
        interaction_index = button_info['index']
        
        removed_list = removed_interactions or []
        if interaction_index not in removed_list:
            removed_list.append(interaction_index)
        
        return removed_list
    
    return removed_interactions or []

@app.callback(
    Output('llm-results-display', 'children', allow_duplicate=True),
    [Input('removed-llm-interactions', 'data')],
    [State('llm-interactions', 'data'),
     State('llm-provider', 'value'),
     State('target-column', 'value'),
     State('resample-freq', 'value')],
    prevent_initial_call=True
)
def update_llm_display_after_removal(removed_interactions, llm_interactions, provider, target_column, resample_freq):
    if not llm_interactions:
        return []
    
    removed_indices = removed_interactions or []
    active_interactions = [interaction for i, interaction in enumerate(llm_interactions) 
                         if i not in removed_indices]
    
    if not active_interactions:
        return html.Div([
            html.H5([html.I(className='fas fa-info-circle', style={'marginRight': '10px', 'color': '#f59e0b'}), 
                    'All interactions removed'], style={'color': '#f59e0b'}),
            html.P("You have removed all LLM-suggested interactions.", style={'color': '#6b7280'})
        ])
    
    interaction_cards = []
    for i, interaction in enumerate(llm_interactions):
        if i in removed_indices:
            continue
            
        source, target, score, lag = interaction[:4]
        justification = interaction[4] if len(interaction) > 4 else "Literature-based interaction"
        
        if '\n\nReference:' in justification:
            main_text, reference = justification.split('\n\nReference:', 1)
            reference = reference.strip()
        else:
            main_text = justification
            reference = ""
        
        card = html.Div([
            html.Div([
                html.Div([
                    html.Strong(f"{source} → {target}", style={'fontSize': '1.1rem', 'color': '#374151'}),
                    html.Span(f" (Score: {score:.2f}, Lag: {lag})", 
                            style={'color': '#6b7280', 'marginLeft': '10px', 'fontSize': '0.9rem'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
                
                html.P(main_text, style={
                    'fontSize': '0.85rem', 
                    'color': '#4b5563', 
                    'margin': '0 0 8px 0',
                    'fontStyle': 'italic',
                    'lineHeight': '1.4'
                }),
                
                html.Div([
                    html.Strong("Reference: ", style={'fontSize': '0.8rem', 'color': '#374151'}),
                    html.Span(reference, style={
                        'fontSize': '0.8rem', 
                        'color': '#6b7280',
                        'lineHeight': '1.3'
                    })
                ], style={'marginTop': '8px'}) if reference else None
                
            ], style={'flex': '1', 'marginRight': '15px'}),
            
            html.Button([
                html.I(className='fas fa-trash', style={'fontSize': '0.9rem'})
            ], 
            id={'type': 'remove-llm-interaction', 'index': i},
            className='modern-button',
            style={
                'backgroundColor': '#ef4444',
                'padding': '8px 12px',
                'minWidth': '40px',
                'height': '40px'
            },
            title=f"Remove {source} → {target} interaction")
            
        ], style={
            'display': 'flex', 
            'alignItems': 'flex-start', 
            'padding': '15px', 
            'backgroundColor': '#f0fdf4', 
            'margin': '8px 0', 
            'borderRadius': '8px',
            'border': '1px solid #bbf7d0'
        })
        
        interaction_cards.append(card)
    
    return html.Div([
        html.H5([
            html.I(className='fas fa-check-circle', style={'marginRight': '10px', 'color': '#10b981'}), 
            f'{len(active_interactions)} literature-based interactions selected for {target_column or "target"} (resolution: {resample_freq or "unknown"})'
        ], style={'color': '#10b981', 'marginBottom': '15px'}),
        
        html.Div(interaction_cards, style={'maxHeight': '400px', 'overflowY': 'auto'}),
        
        html.P("Click the trash icon to remove unwanted interactions. Remaining interactions will be automatically added to your analysis.", 
               style={'marginTop': '15px', 'color': '#059669', 'fontStyle': 'italic', 'fontSize': '0.9rem'})
    ])

@app.callback(
    [Output('known-interactions-section', 'style'),
     Output({'type': 'known-x1', 'index': ALL}, 'options'),
     Output({'type': 'known-x2', 'index': ALL}, 'options')],
    [Input('target-column', 'value'),
     Input('stored-columns', 'data')],
    [State({'type': 'known-x1', 'index': ALL}, 'id'),
     State({'type': 'known-x2', 'index': ALL}, 'id')]
)
def show_known_interactions_section(target_column, columns, x1_ids, x2_ids):
    num_x1_components = len(x1_ids) if x1_ids else 1
    num_x2_components = len(x2_ids) if x2_ids else 1
    
    if columns and target_column:
        variable_options = [{'label': col, 'value': col} for col in columns if col.lower() not in ['date', 'index']]
        x1_options = [variable_options] * num_x1_components
        x2_options = [variable_options] * num_x2_components
        return ({'display': 'block'}, x1_options, x2_options)
    else:
        x1_options = [[]] * num_x1_components
        x2_options = [[]] * num_x2_components
        return ({'display': 'none'}, x1_options, x2_options)

@app.callback(
    Output('known-interactions-container', 'children'),
    [Input('add-known-interaction', 'n_clicks'),
     Input({'type': 'remove-known', 'index': ALL}, 'n_clicks')],
    [State('known-interactions-container', 'children'),
     State('stored-columns', 'data')]
)
def manage_known_interactions(add_clicks, remove_clicks, current_children, columns):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_children
    
    trigger = ctx.triggered[0]
    prop_id = trigger['prop_id']
    
    if not columns:
        return current_children
    
    variable_options = [{'label': col, 'value': col} for col in columns if col.lower() not in ['date', 'index']]
    
    if 'add-known-interaction' in prop_id:
        new_index = len(current_children)
        new_interaction = html.Div(id=f'known-interaction-{new_index}', children=[
            html.Div([
                html.Div([
                    html.Label('Source Variable', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                    dcc.Dropdown(id={'type': 'known-x1', 'index': new_index}, options=variable_options, className='input-field', style={'height': '46px'}),
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label('Target Variable', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                    dcc.Dropdown(id={'type': 'known-x2', 'index': new_index}, options=variable_options, className='input-field', style={'height': '46px'}),
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label('Score (0-1)', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                    dcc.Input(id={'type': 'known-score', 'index': new_index}, type='number', min=0, max=1, step=0.1, value=0.8, className='input-field', style={'height': '46px'}),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label('Time Lag', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                    dcc.Input(id={'type': 'known-lag', 'index': new_index}, type='number', min=0, step=1, value=0, className='input-field', style={'height': '46px'}),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label('', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                    html.Button([html.I(className='fas fa-trash')], 
                               id={'type': 'remove-known', 'index': new_index}, 
                               className='modern-button', 
                               style={'backgroundColor': '#ef4444', 'padding': '10px 15px', 'height': '46px', 'width': '100%'}),
                ], style={'width': '12%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'top'})
            ], style={'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '10px'})
        ])
        return current_children + [new_interaction]
    
    elif 'remove-known' in prop_id:
        import json
        button_info = json.loads(prop_id.split('.')[0])
        remove_index = button_info['index']
        
        new_children = []
        new_index = 0
        for i, child in enumerate(current_children):
            if i != remove_index:
                new_interaction = html.Div(id=f'known-interaction-{new_index}', children=[
                    html.Div([
                        html.Div([
                            html.Label('Source Variable', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                            dcc.Dropdown(id={'type': 'known-x1', 'index': new_index}, options=variable_options, className='input-field', style={'height': '46px'}),
                        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.Label('Target Variable', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                            dcc.Dropdown(id={'type': 'known-x2', 'index': new_index}, options=variable_options, className='input-field', style={'height': '46px'}),
                        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.Label('Score (0-1)', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                            dcc.Input(id={'type': 'known-score', 'index': new_index}, type='number', min=0, max=1, step=0.1, value=0.8, className='input-field', style={'height': '46px'}),
                        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.Label('Time Lag', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                            dcc.Input(id={'type': 'known-lag', 'index': new_index}, type='number', min=0, step=1, value=0, className='input-field', style={'height': '46px'}),
                        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.Label('', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'height': '20px'}),
                            html.Button([html.I(className='fas fa-trash')], 
                                       id={'type': 'remove-known', 'index': new_index}, 
                                       className='modern-button', 
                                       style={'backgroundColor': '#ef4444', 'padding': '10px 15px', 'height': '46px', 'width': '100%'}),
                        ], style={'width': '12%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '10px'})
                ])
                new_children.append(new_interaction)
                new_index += 1
        
        return new_children if new_children else [html.Div("No known interactions added yet.", style={'textAlign': 'center', 'color': '#6b7280', 'padding': '20px'})]
    
    return current_children

@app.callback(
    Output('known-interactions', 'data'),
    [Input({'type': 'known-x1', 'index': ALL}, 'value'),
     Input({'type': 'known-x2', 'index': ALL}, 'value'),
     Input({'type': 'known-score', 'index': ALL}, 'value'),
     Input({'type': 'known-lag', 'index': ALL}, 'value'),
     Input('llm-interactions', 'data'),
     Input('removed-llm-interactions', 'data')]
)
def store_known_interactions(x1_values, x2_values, score_values, lag_values, llm_interactions, removed_llm_interactions):
    manual_interactions = []
    for x1, x2, score, lag in zip(x1_values, x2_values, score_values, lag_values):
        if x1 and x2 and x1 != x2:
            score = float(score) if score is not None and 0 <= float(score) <= 1 else 0.8
            lag = int(lag) if lag is not None and lag >= 0 else 0
            manual_interactions.append([x1, x2, score, lag])
    
    all_interactions = manual_interactions.copy()
    
    if llm_interactions:
        removed_indices = removed_llm_interactions or []
        
        for i, llm_interaction in enumerate(llm_interactions):
            if i in removed_indices:
                continue
                
            llm_interaction_clean = llm_interaction[:4]
            
            pair_exists = any(
                (manual[0] == llm_interaction_clean[0] and manual[1] == llm_interaction_clean[1]) 
                for manual in manual_interactions
            )
            if not pair_exists:
                all_interactions.append(llm_interaction_clean)
    
    return all_interactions

@app.callback(
    Output('stored-output-folder', 'data'),
    Input('upload-data', 'children'),
    prevent_initial_call=False
)
def create_output_folder_on_load(_):
    d = datetime.datetime.today()
    output_folder = os.path.join(os.getcwd(), "Results", f"{d.year}{d.month}{d.day}{d.hour}{d.minute}{d.second}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('target-column', 'options'),
     Output('confounder-columns', 'options'),
     Output('stored-filename', 'data'),
     Output('stored-columns', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            df.to_csv(temp_file.name, index=False)
            temp_file.close()

            columns_options = [{'label': col, 'value': col} for col in df.columns if col.lower() != 'date']
            columns = df.columns.tolist()
            
            return (
                html.Div([
                    html.Div([
                        html.I(className='fas fa-check-circle', style={'color': '#10b981', 'fontSize': '1.5rem', 'marginRight': '10px'}),
                        html.H4(f'Successfully uploaded: {filename}', style={'margin': '0', 'color': '#10b981', 'display': 'inline'})
                    ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#d1fae5', 'borderRadius': '10px', 'display': 'flex', 'alignItems': 'center'}),
                    
                    html.H5([html.I(className='fas fa-table', style={'marginRight': '10px'}), 'Data Preview'], 
                            style={'color': '#4f46e5', 'marginBottom': '15px'}),
                    
                    html.Div([
                        html.P(f"Rows: {len(df):,} | Columns: {len(df.columns)}", 
                               style={'color': '#6b7280', 'marginBottom': '15px', 'fontWeight': '500'}),
                        
                        dash_table.DataTable(
                            data=df.head(10).to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in df.columns],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '12px',
                                'fontFamily': 'Inter',
                                'fontSize': '0.9rem',
                                'border': '1px solid #e5e7eb'
                            },
                            style_header={
                                'backgroundColor': '#f8fafc',
                                'fontWeight': '600',
                                'color': '#374151',
                                'border': '1px solid #d1d5db'
                            },
                            style_data={
                                'backgroundColor': 'white',
                                'color': '#374151'
                            }
                        )
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'border': '1px solid #e5e7eb'})
                ]),
                columns_options,
                columns_options,
                temp_file.name,
                columns
            )
        except Exception as e:
            return (
                html.Div([
                    html.I(className='fas fa-exclamation-triangle', style={'color': '#ef4444', 'fontSize': '1.5rem', 'marginRight': '10px'}),
                    html.H4('File Upload Error', style={'margin': '0', 'color': '#ef4444', 'display': 'inline'}),
                    html.P(f'Error: {str(e)}', style={'marginTop': '10px', 'color': '#6b7280'})
                ], style={'padding': '20px', 'backgroundColor': '#fee2e2', 'borderRadius': '10px', 'border': '1px solid #fca5a5'}),
                [], [], None, []
            )
    return (
        html.Div([
            html.I(className='fas fa-cloud-upload-alt', style={'fontSize': '3rem', 'color': '#9ca3af', 'marginBottom': '15px'}),
            html.P('Upload a CSV file to begin analysis', style={'color': '#6b7280', 'fontSize': '1.1rem'})
        ], style={'textAlign': 'center', 'padding': '40px', 'color': '#9ca3af'}), 
        [], [], None, []
    )

@app.callback(
    Output('upload-categorization-file', 'style'),
    [Input('categorization-mode', 'value')]
)
def toggle_categorization_input(value):
    if value == 'upload':
        return {'display': 'block', 'marginBottom': '20px'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('output-categories-upload', 'children'),
     Output('categorization-filepath', 'data')],
    [Input('upload-categories-file', 'contents')],
    [State('upload-categories-file', 'filename')]
)
def update_categories_file(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            df.to_csv(temp_file.name, index=False)
            temp_file.close()

            return (
                html.Div([
                    html.H5(filename),
                    html.H6("Preview of the first 10 rows:"),
                    dash_table.DataTable(
                        data=df.head(10).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df.columns]
                    )
                ]),
                temp_file.name
            )
        except Exception as e:
            return html.Div([
                'There was an error processing this file: ' + str(e)
            ]), None
    return html.Div('Upload CSV file to see a preview.'), None

@app.callback(
    [Output('table-editing', 'columns'),
     Output('table-editing', 'data'),
     Output('table-editing', 'dropdown')],
    [Input('first-step-completed', 'data')],
    [State('stored-output-folder', 'data')]
)
def load_curated_file(first_step_completed, output_folder):
    print(f"DEBUG: load_curated_file called with first_step_completed={first_step_completed}, output_folder={output_folder}")
    
    if first_step_completed and output_folder:
        try:
            
            curated_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
            manual_edits_flag = os.path.join(output_folder, 'MANUAL_EDITS_MADE.flag')
            
            if not os.path.exists(curated_file):
                print("DEBUG: CSV file doesn't exist")
                return [], [], {}
                
            df = pd.read_csv(curated_file)
            print(f"DEBUG: Loaded CSV with {len(df)} rows")
            
            # ALWAYS preserve manual edits if flag exists
            if os.path.exists(manual_edits_flag):
                print("DEBUG: Manual edits flag detected - preserving user changes")
                # Don't let anything overwrite this data
            
            # Convert is_Valid column to integers
            if 'is_Valid' in df.columns:
                print(f"DEBUG: Original is_Valid values: {df['is_Valid'].tolist()}")
                df['is_Valid'] = pd.to_numeric(df['is_Valid'], errors='coerce').fillna(2).astype(int)
                print(f"DEBUG: Converted is_Valid values: {df['is_Valid'].tolist()}")
                print(f"DEBUG: Value counts: {df['is_Valid'].value_counts().to_dict()}")
            
            columns = []
            for col in df.columns:
                col_config = {"name": col, "id": col, "editable": True}
                if col == 'is_Valid':
                    col_config["presentation"] = "dropdown"
                elif col in ['Score', 'timeToEffect']:
                    col_config["type"] = "numeric"
                columns.append(col_config)
            
            dropdown_options = {
                'is_Valid': {
                    'options': [
                        {'label': 'Invalid', 'value': 0},
                        {'label': 'Under Review', 'value': 1}, 
                        {'label': 'Valid', 'value': 2}
                    ]
                }
            }
            
            data = df.to_dict('records')
            print(f"DEBUG: Returning {len(data)} rows to table")
            
            with open('debug_log.txt', 'a') as f:
                f.write(f"\n=== TABLE LOADING DEBUG ===\n")
                f.write(f"Loading table with {len(data)} rows\n")
                f.write(f"Sample is_Valid from table: {[row.get('is_Valid') for row in data[:5]]}\n")
            
            return columns, data, dropdown_options
            
        except Exception as e:
            print(f"DEBUG: Error in load_curated_file: {e}")
            return [], [], {}
    
    return [], [], {}


@app.callback(
    [Output('save-confirmation', 'children'),
     Output('table-editing', 'dropdown', allow_duplicate=True)],
    [Input('save-edits', 'n_clicks')],
    [State('table-editing', 'data'),
     State('table-editing', 'columns'),
     State('stored-output-folder', 'data')],
    prevent_initial_call=True
)
def save_edited_table(n_clicks, rows, columns, output_folder):
    dropdown_options = {
        'is_Valid': {
            'options': [
                {'label': 'Invalid', 'value': 0},
                {'label': 'Under Review', 'value': 1}, 
                {'label': 'Valid', 'value': 2}
            ]
        }
    }
    
    if n_clicks > 0 and output_folder and rows and columns:
        try:
            df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
            
            df = df.dropna(subset=['x1', 'x2'])
            df = df[df['x1'].str.strip() != '']
            df = df[df['x2'].str.strip() != '']
            
            if 'Score' in df.columns:
                df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0.0)
            if 'timeToEffect' in df.columns:
                df['timeToEffect'] = pd.to_numeric(df['timeToEffect'], errors='coerce').fillna(0)
            if 'is_Valid' in df.columns:
                df['is_Valid'] = pd.to_numeric(df['is_Valid'], errors='coerce').fillna(1)
            
            d = datetime.datetime.today()
            output_filepath = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
            
            if os.path.exists(output_filepath):
                output_filepath_archived = os.path.join(output_folder, "CCM_ECCM_curated_"+ f"{d.year}{d.month}{d.day}{d.hour}{d.minute}{d.second}"+".csv")
                df_previous = pd.read_csv(output_filepath)
                df_previous.to_csv(output_filepath_archived, index=False)
            
            df.to_csv(output_filepath, index=False)
            
            # Create flag to prevent future overwrites
            with open(os.path.join(output_folder, 'MANUAL_EDITS_MADE.flag'), 'w') as f:
                f.write('User made manual edits - do not overwrite')
            
            
            df_results = pd.read_csv(output_filepath)
            visualize_network(df_results, os.path.join(output_folder, 'network_plot.png'))
            shutil.copy(os.path.join(output_folder, 'network_plot.png'), os.path.join('assets', 'network_plot.png'))        
            
            timestamp = str(int(datetime.datetime.now().timestamp()))
            
            return (html.Div([
                html.P("Edits saved successfully!", style={'color': 'green', 'fontWeight': 'bold'}),
                html.P(f"Saved {len(df)} connections. Network plot updated.", style={'color': 'green'}),
                html.Div(style={'textAlign': 'center', 'marginTop': '20px'}, children=[
                    html.H5('Updated Network:', style={'marginBottom': '10px'}),
                    html.Img(src=f'assets/network_plot.png?t={timestamp}', style={'maxWidth': '100%', 'height': 'auto', 'borderRadius': '10px', 'border': '2px solid #10b981'})
                ])
            ], style={'marginTop': '20px'}), dropdown_options)
        except Exception as e:
            return (html.Div(f"❌ Failed to save edits: {str(e)}", style={'color': 'red', 'marginTop': '20px'}), dropdown_options)
    
    return ('', dropdown_options)

@app.callback(
    [Output('table-editing', 'data', allow_duplicate=True)],
    [Input('add-row-button', 'n_clicks')],
    [State('table-editing', 'data'),
     State('table-editing', 'columns')],
    prevent_initial_call=True
)
def add_new_row(n_clicks, current_data, columns):
    if n_clicks > 0 and columns:
        new_row = {}
        for col in columns:
            col_id = col['id']
            if col_id == 'x1':
                new_row[col_id] = 'Variable1'
            elif col_id == 'x2':
                new_row[col_id] = 'Variable2'
            elif col_id == 'Score':
                new_row[col_id] = 0.0
            elif col_id == 'timeToEffect':
                new_row[col_id] = 0
            elif col_id == 'is_Valid':
                new_row[col_id] = 2
            else:
                new_row[col_id] = ''
        
        updated_data = current_data + [new_row] if current_data else [new_row]
        return [updated_data]
    
    return [current_data] if current_data else [[]]

@app.callback(
    [Output('model-stats-section', 'children'),
     Output('model-stats-section', 'style'),
     Output('inference-section', 'children'),
     Output('inference-section', 'style'),
     Output('inference-model-data', 'data')],
    [Input('fourth-step', 'style')],
    [State('stored-output-folder', 'data')]
)
def auto_load_model_files(fourth_step_style, output_folder):
    if fourth_step_style.get('display') != 'block' or not output_folder:
        return [], {'display': 'none'}, [], {'display': 'none'}, {}
    
    try:
        bnlearn_model_path = os.path.join(output_folder, 'bnlearn_model.pkl')
        dict_essentials_path = os.path.join(output_folder, 'dict_model_essentials.pickle')
        scenario_data_path = os.path.join(output_folder, 'scenario_data.pickle')
        
        if not os.path.exists(bnlearn_model_path) or not os.path.exists(dict_essentials_path):
            return ([html.Div([
                html.I(className='fas fa-exclamation-triangle', style={'color': '#f59e0b', 'fontSize': '1.5rem', 'marginRight': '10px'}),
                html.Span('Model files not found. Please ensure the Bayesian Network step completed successfully.', 
                         style={'color': '#f59e0b'})
            ], style={'padding': '15px', 'backgroundColor': '#fffbeb', 'borderRadius': '10px', 'border': '1px solid #fbbf24'})], 
            {'display': 'block'}, [], {'display': 'none'}, {})
        
        with open(dict_essentials_path, 'rb') as f:
            dict_model_essentials = pickle.load(f)
        
        scenario_data = {}
        if os.path.exists(scenario_data_path):
            with open(scenario_data_path, 'rb') as f:
                scenario_data = pickle.load(f)
        
        stats_section = html.Div(className='parameter-card', children=[
            html.H4([html.I(className='fas fa-chart-pie', style={'marginRight': '10px'}), 'Model Statistics'], className='card-title'),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className='fas fa-bullseye', style={'fontSize': '2rem', 'color': '#4f46e5', 'marginBottom': '10px'}),
                        html.H5('Target Variable', style={'margin': '0', 'color': '#374151'}),
                        html.P(dict_model_essentials['target'], style={'fontSize': '1.2rem', 'fontWeight': '600', 'margin': '5px 0 0 0', 'color': '#4f46e5'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8fafc', 'borderRadius': '10px'}),
                ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div([
                        html.I(className='fas fa-network-wired', style={'fontSize': '2rem', 'color': '#10b981', 'marginBottom': '10px'}),
                        html.H5('Model Nodes', style={'margin': '0', 'color': '#374151'}),
                        html.P(str(len(dict_model_essentials['nodes'])), style={'fontSize': '1.2rem', 'fontWeight': '600', 'margin': '5px 0 0 0', 'color': '#10b981'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f0fdf4', 'borderRadius': '10px'}),
                ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div([
                        html.I(className='fas fa-check-circle', style={'fontSize': '2rem', 'color': '#f59e0b', 'marginBottom': '10px'}),
                        html.H5('Accuracy', style={'margin': '0', 'color': '#374151'}),
                        html.P(f"{dict_model_essentials['accuracy']:.3f}", style={'fontSize': '1.2rem', 'fontWeight': '600', 'margin': '5px 0 0 0', 'color': '#f59e0b'})
                    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#fffbeb', 'borderRadius': '10px'}),
                ], style={'width': '32%', 'display': 'inline-block'}),
            ])
        ])
        
        input_fields = []
        for i, node in enumerate(dict_model_essentials['nodes']):
            if node != dict_model_essentials['target']:
                input_fields.append(
                    html.Div([
                        html.Label(node, style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block', 'color': '#374151'}),
                        dcc.Dropdown(
                            id={'type': 'inference-input', 'index': node},
                            options=[
                                {'label': 'Low (0)', 'value': 0},
                                {'label': 'Medium (1)', 'value': 1},
                                {'label': 'High (2)', 'value': 2}
                            ],
                            value=1,
                            className='input-field',
                            style={'marginBottom': '10px'}
                        )
                    ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%', 'verticalAlign': 'top'})
                )
        
        scenario_type_options = [{'label': 'Custom Input', 'value': 'custom'}]
        if scenario_data.get('high_scenarios'):
            scenario_type_options.append({'label': f'High {dict_model_essentials["target"]} Scenarios ({len(scenario_data["high_scenarios"])})', 'value': 'high'})
        if scenario_data.get('low_scenarios'):
            scenario_type_options.append({'label': f'Low {dict_model_essentials["target"]} Scenarios ({len(scenario_data["low_scenarios"])})', 'value': 'low'})
        
        inference_section = html.Div(className='parameter-card', children=[
            html.H4([html.I(className='fas fa-magic', style={'marginRight': '10px'}), 'Run Prediction'], className='card-title'),
            
            html.Div(className='parameter-card', style={'backgroundColor': '#f0f9ff', 'border': '1px solid #0ea5e9', 'marginBottom': '25px'}, children=[
                html.H5([html.I(className='fas fa-lightbulb', style={'marginRight': '10px'}), 'Quick Scenarios'], 
                        style={'color': '#0ea5e9', 'marginBottom': '15px'}),
                
                html.Div([
                    html.Div([
                        html.Label('Scenario Type:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='scenario-type-dropdown',
                            options=scenario_type_options,
                            value='custom',
                            className='input-field',
                            style={'height': '50px'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.Label('Specific Scenario:', style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='specific-scenario-dropdown',
                            options=[],
                            value=None,
                            className='input-field',
                            style={'height': '50px'},
                            disabled=True
                        )
                    ], style={'width': '48%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                
                html.Div(style={'textAlign': 'center'}, children=[
                    html.Button([
                        html.I(className='fas fa-magic', style={'marginRight': '10px'}),
                        'Apply Selected Scenario'
                    ], id='apply-scenario-btn', n_clicks=0, className='modern-button secondary-button', disabled=True)
                ])
            ]),
            
            html.P(f"Set input values for each variable to predict the probability of {dict_model_essentials['target']} outcomes.", 
                   style={'marginBottom': '25px', 'color': '#6b7280'}),
            
            html.Div(input_fields, style={'marginBottom': '25px'}),
            
            html.Div(style={'textAlign': 'center'}, children=[
                html.Button([
                    html.I(className='fas fa-play', style={'marginRight': '10px'}),
                    'Run Inference'
                ], id='run-inference-btn', n_clicks=0, className='modern-button success-button', style={'marginBottom': '20px'})
            ]),
            
            html.Div(id='inference-results', style={'marginTop': '20px'})
        ])
        
        return (stats_section, {'display': 'block'}, 
                inference_section, {'display': 'block'}, 
                {'dict_model_essentials': dict_model_essentials, 'scenario_data': scenario_data})
        
    except Exception as e:
        error_msg = html.Div([
            html.I(className='fas fa-exclamation-triangle', style={'color': '#ef4444', 'fontSize': '1.5rem', 'marginRight': '10px'}),
            html.Span(f'Error loading model files: {str(e)}', style={'color': '#ef4444'})
        ], style={'padding': '15px', 'backgroundColor': '#fee2e2', 'borderRadius': '10px', 'border': '1px solid #fca5a5'})
        
        return error_msg, {'display': 'block'}, [], {'display': 'none'}, {}

@app.callback(
    Output('inference-results', 'children'),
    [Input('run-inference-btn', 'n_clicks')],
    [State('inference-model-data', 'data'), 
     State({'type': 'inference-input', 'index': ALL}, 'value'),
     State({'type': 'inference-input', 'index': ALL}, 'id'),
     State('stored-output-folder', 'data')]
)
def run_inference(n_clicks, model_data, values, input_ids, output_folder):
    if n_clicks == 0 or not model_data:
        return ''
    
    try:
        dict_model_essentials = model_data.get('dict_model_essentials')
        if not dict_model_essentials:
            raise Exception("Model essentials not available")
        
        evidence = {}
        for value, input_id in zip(values, input_ids):
            node_name = input_id['index']
            if value is not None:
                evidence[node_name] = int(value)
        
        if not output_folder:
            raise Exception("No output folder found in session")
        
        model_path = os.path.join(output_folder, 'bnlearn_model.pkl')
        
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            local_bn_model = pickle.load(f)
                
        target_var = dict_model_essentials['target']
        
        evidence_clean = {}
        for k, v in evidence.items():
            if v is not None:
                try:
                    evidence_clean[k] = str(int(float(v)))
                except (ValueError, TypeError):
                    evidence_clean[k] = '1'  
        

        q1 = bn.inference.fit(local_bn_model, variables=[target_var], evidence=evidence_clean)
        
        prediction_low = q1.df.p[0]
        prediction_high = q1.df.p[1]
        
        
        return html.Div([
            html.H5([html.I(className='fas fa-chart-bar', style={'marginRight': '10px'}), 'Prediction Results'], 
                    style={'color': '#4f46e5', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className='fas fa-arrow-down', style={'fontSize': '2.5rem', 'color': '#ef4444', 'marginBottom': '15px'}),
                        html.H4('Low Probability', style={'margin': '0', 'color': '#374151', 'marginBottom': '10px'}),
                        html.P(f"{prediction_low:.1%}", style={'fontSize': '2rem', 'fontWeight': '700', 'margin': '0', 'color': '#ef4444'}),
                        html.P(f"({prediction_low:.4f})", style={'fontSize': '0.9rem', 'color': '#6b7280', 'margin': '5px 0 0 0'})
                    ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#fef2f2', 'borderRadius': '15px', 'border': '2px solid #fca5a5'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                html.Div([
                    html.Div([
                        html.I(className='fas fa-arrow-up', style={'fontSize': '2.5rem', 'color': '#10b981', 'marginBottom': '15px'}),
                        html.H4('High Probability', style={'margin': '0', 'color': '#374151', 'marginBottom': '10px'}),
                        html.P(f"{prediction_high:.1%}", style={'fontSize': '2rem', 'fontWeight': '700', 'margin': '0', 'color': '#10b981'}),
                        html.P(f"({prediction_high:.4f})", style={'fontSize': '0.9rem', 'color': '#6b7280', 'margin': '5px 0 0 0'})
                    ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#f0fdf4', 'borderRadius': '15px', 'border': '2px solid #a7f3d0'})
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.H6('Input Evidence:', style={'fontWeight': '600', 'marginBottom': '10px', 'color': '#374151'}),
                html.Div(
                    ' | '.join([f"{k}: {['Low', 'Medium', 'High'][v]}" for k, v in evidence.items()]),
                    style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '10px', 'fontSize': '0.95rem'}
                )
            ])
        ])
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return html.Div([
            html.I(className='fas fa-exclamation-triangle', style={'color': '#ef4444', 'fontSize': '1.5rem', 'marginRight': '10px'}),
            html.Span(f'Error during inference: {str(e)}', style={'color': '#ef4444'})
        ], style={'padding': '15px', 'backgroundColor': '#fee2e2', 'borderRadius': '10px', 'border': '1px solid #fca5a5'})

@app.callback(
    Output('download-tal-paper', 'data'),
    Input('download-tal-paper-link', 'n_clicks'),
    prevent_initial_call=True
)
def download_tal_paper(n_clicks):
    if n_clicks:
        try:
            paper_file_path = 'docs/tal_2024.pdf'
            if os.path.exists(paper_file_path):
                return dcc.send_file(paper_file_path, filename="Tal_et_al_2024.pdf")
            else:
                content = "Tal et al. 2024 paper not found. Please contact the administrator."
                return dict(content=content, filename="tal_paper_not_found.txt")
        except Exception as e:
            content = f"Error downloading paper: {str(e)}"
            return dict(content=content, filename="download_error.txt")
    return None

@app.callback(
    Output('download-instructions', 'data'),
    Input('download-instructions-link', 'n_clicks'),
    prevent_initial_call=True
)
def download_instructions(n_clicks):
    if n_clicks:
        try:
            instructions_file_path = 'docs/instructions.txt'
            if os.path.exists(instructions_file_path):
                with open(instructions_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return dict(content=content, filename="CEcBaN_Instructions.txt")
            else:
                content = """CEcBaN: CCM ECCM Bayesian Network Analysis Tool
User Instructions

CEcBaN implements the causal discovery methodology from Tal et al. (2024).

WORKFLOW:
1. Upload CSV data with time series
2. Configure analysis parameters  
3. Run CCM-ECCM analysis
4. Refine network connections
5. Build Bayesian Network model
6. Run predictions

PARAMETERS:
- Target Column: Main variable to predict
- Confounders: External driving variables
- Subset Length: Window size for analysis (30-100 recommended)
- Embedding Dimension: Phase space reconstruction (2-4 typical)
- ECCM Lag Strategy: Prefer immediate vs strongest effects

REFERENCES:
Tal, O., Ostrovsky, I., & Gal, G. (2024). A framework for identifying factors controlling cyanobacterium Microcystis flos‐aquae blooms by coupled CCM–ECCM Bayesian networks. Ecology and Evolution, 14(6), e11475.

Contact: Tal Lab, Israel Oceanographic and Limnological Research"""
                return dict(content=content, filename="CEcBaN_Instructions.txt")
        except Exception as e:
            content = f"Error downloading instructions: {str(e)}"
            return dict(content=content, filename="download_error.txt")
    return None

@app.callback(
    Output('download-categories-format', 'data'),
    Input('download-categories-format-link', 'n_clicks'),
    prevent_initial_call=True
)
def download_categories_format(n_clicks):
    if n_clicks:
        try:
            categories_file_path = 'docs/categories.txt'
            if os.path.exists(categories_file_path):
                with open(categories_file_path, 'r') as f:
                    content = f.read()
            else:
                content = """variable,bins
Temperature,0;0.3;0.7;1
Humidity,0;0.25;0.5;0.75;1
Wind_Speed,0;0.4;0.8;1
Pressure,0;0.33;0.66;1"""
            
            return dict(content=content, filename="categorization_format.csv")
        except Exception as e:
            content = """variable,bins
Temperature,0;0.3;0.7;1
Humidity,0;0.25;0.5;0.75;1
Wind_Speed,0;0.4;0.8;1
Pressure,0;0.33;0.66;1"""
            return dict(content=content, filename="categorization_format.csv")
    return None

@app.callback(
    [Output('specific-scenario-dropdown', 'options'),
     Output('specific-scenario-dropdown', 'disabled'),
     Output('apply-scenario-btn', 'disabled')],
    [Input('scenario-type-dropdown', 'value')],
    [State('inference-model-data', 'data')]
)
def update_specific_scenarios(scenario_type, model_data):
    if not model_data or scenario_type == 'custom':
        return [], True, True
    
    scenario_data = model_data.get('scenario_data', {})
    
    if scenario_type == 'high':
        scenarios = scenario_data.get('high_scenarios', [])
        frequencies = scenario_data.get('high_frequencies', [])
    elif scenario_type == 'low':
        scenarios = scenario_data.get('low_scenarios', [])
        frequencies = scenario_data.get('low_frequencies', [])
    else:
        return [], True, True
    
    if not scenarios or not frequencies:
        return [], True, True
    
    scenario_freq_pairs = list(zip(scenarios, frequencies, range(len(scenarios))))
    scenario_freq_pairs.sort(key=lambda x: x[1], reverse=True) 
    
    options = []
    for sorted_idx, (scenario, frequency, original_idx) in enumerate(scenario_freq_pairs):
        freq_text = f" ({frequency:.1f}%)"
        scenario_label = f'High Scenario' if scenario_type == 'high' else f'Low Scenario'
        options.append({
            'label': f'{scenario_label} {sorted_idx+1}{freq_text}', 
            'value': original_idx  
        })
    
    return options, len(options) == 0, len(options) == 0

@app.callback(
    Output('inference-results', 'children', allow_duplicate=True),
    [Input('run-inference-btn', 'n_clicks'),
     Input('specific-scenario-dropdown', 'value')],
    [State('inference-model-data', 'data'), 
     State({'type': 'inference-input', 'index': ALL}, 'value'),
     State({'type': 'inference-input', 'index': ALL}, 'id'),
     State('stored-output-folder', 'data'),
     State('scenario-type-dropdown', 'value')],
    prevent_initial_call=True
)
def run_inference_with_scenario_image(n_clicks_inference, selected_scenario_index, model_data, values, input_ids, output_folder, scenario_type):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
 
    
    if n_clicks_inference > 0 and ctx.triggered:
        if not model_data:
            return ''
        
        try:
            dict_model_essentials = model_data.get('dict_model_essentials')
            if not dict_model_essentials:
                raise Exception("Model essentials not available")
            
            evidence = {}
            for value, input_id in zip(values, input_ids):
                node_name = input_id['index']
                if value is not None:
                    evidence[node_name] = int(value)
            
            if not output_folder:
                raise Exception("No output folder found in session")
            
            model_path = os.path.join(output_folder, 'bnlearn_model.pkl')
            
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                local_bn_model = pickle.load(f)
            
            target_var = dict_model_essentials['target']
            
            evidence_clean = {}
            for k, v in evidence.items():
                if v is not None:
                    try:
                        evidence_clean[k] = str(int(float(v)))
                    except (ValueError, TypeError):
                        evidence_clean[k] = '1'  
            
            q1 = bn.inference.fit(local_bn_model, variables=[target_var], evidence=evidence_clean)
            
            prediction_low = q1.df.p[0]
            prediction_high = q1.df.p[1]
            
            results_content = [
                html.H5([html.I(className='fas fa-chart-bar', style={'marginRight': '10px'}), 'Prediction Results'], 
                        style={'color': '#4f46e5', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Div([
                            html.I(className='fas fa-arrow-down', style={'fontSize': '2.5rem', 'color': '#ef4444', 'marginBottom': '15px'}),
                            html.H4('Low Probability', style={'margin': '0', 'color': '#374151', 'marginBottom': '10px'}),
                            html.P(f"{prediction_low:.1%}", style={'fontSize': '2rem', 'fontWeight': '700', 'margin': '0', 'color': '#ef4444'}),
                            html.P(f"({prediction_low:.4f})", style={'fontSize': '0.9rem', 'color': '#6b7280', 'margin': '5px 0 0 0'})
                        ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#fef2f2', 'borderRadius': '15px', 'border': '2px solid #fca5a5'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.Div([
                            html.I(className='fas fa-arrow-up', style={'fontSize': '2.5rem', 'color': '#10b981', 'marginBottom': '15px'}),
                            html.H4('High Probability', style={'margin': '0', 'color': '#374151', 'marginBottom': '10px'}),
                            html.P(f"{prediction_high:.1%}", style={'fontSize': '2rem', 'fontWeight': '700', 'margin': '0', 'color': '#10b981'}),
                            html.P(f"({prediction_high:.4f})", style={'fontSize': '0.9rem', 'color': '#6b7280', 'margin': '5px 0 0 0'})
                        ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#f0fdf4', 'borderRadius': '15px', 'border': '2px solid #a7f3d0'})
                    ], style={'width': '48%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H6('Input Evidence:', style={'fontWeight': '600', 'marginBottom': '10px', 'color': '#374151'}),
                    html.Div(
                        ' | '.join([f"{k}: {['Low', 'Medium', 'High'][v]}" for k, v in evidence.items()]),
                        style={'padding': '15px', 'backgroundColor': '#f8fafc', 'borderRadius': '10px', 'fontSize': '0.95rem'}
                    )
                ])
            ]
            
            if selected_scenario_index is not None and scenario_type in ['high', 'low'] and output_folder:
                scenario_prefix = 'high' if scenario_type == 'high' else 'low'
                scenario_image_path = os.path.join(output_folder, f"scenario_{scenario_prefix}_{selected_scenario_index:03d}.png")
                
                if os.path.exists(scenario_image_path):
                    assets_image_path = f"scenario_{scenario_prefix}_{selected_scenario_index:03d}.png"
                    shutil.copy(scenario_image_path, os.path.join('assets', assets_image_path))
                    
                    timestamp = str(int(datetime.datetime.now().timestamp()))
                    
                    results_content.append(
                        html.Div([
                            html.H6('Selected Scenario Network:', style={'fontWeight': '600', 'marginTop': '30px', 'marginBottom': '15px', 'color': '#374151'}),
                            html.Div([
                                html.Img(src=f'/assets/{assets_image_path}?t={timestamp}', 
                                        style={'width': '100%', 'height': 'auto', 'borderRadius': '10px', 'border': '2px solid #4f46e5'})
                            ], style={'textAlign': 'center'})
                        ])
                    )
            
            return html.Div(results_content)
            
        except Exception as e:
            import traceback
            print(f"Full error details: {e}")
            traceback.print_exc()
            
            return html.Div([
                html.I(className='fas fa-exclamation-triangle', style={'color': '#ef4444', 'fontSize': '1.5rem', 'marginRight': '10px'}),
                html.Span(f'Error during inference: {str(e)}', style={'color': '#ef4444'})
            ], style={'padding': '15px', 'backgroundColor': '#fee2e2', 'borderRadius': '10px', 'border': '1px solid #fca5a5'})
    
    return ''

@app.callback(
    [Output({'type': 'inference-input', 'index': ALL}, 'value')],
    [Input('apply-scenario-btn', 'n_clicks')],
    [State('scenario-type-dropdown', 'value'),
     State('specific-scenario-dropdown', 'value'),
     State('inference-model-data', 'data'),
     State({'type': 'inference-input', 'index': ALL}, 'id')]
)
def apply_scenario(n_clicks, scenario_type, scenario_index, model_data, input_ids):
    if n_clicks == 0 or not model_data or scenario_type == 'custom' or scenario_index is None:
        raise PreventUpdate
    
    scenario_data = model_data.get('scenario_data', {})
    
    if scenario_type == 'high':
        scenarios = scenario_data.get('high_scenarios', [])
    elif scenario_type == 'low':
        scenarios = scenario_data.get('low_scenarios', [])
    else:
        raise PreventUpdate
    
    if scenario_index >= len(scenarios):
        raise PreventUpdate
    
    selected_scenario = scenarios[scenario_index]
    
    values = []
    for input_id in input_ids:
        node_name = input_id['index']
        values.append(selected_scenario.get(node_name, 1))  
    return [values]

# MAIN PIPELINE CALLBACK WITH ALL THE FUNCTIONALITY
@app.callback(
    [Output('pipeline-output', 'children'),
     Output('loading-output-pipeline', 'children'),
     Output('loading-output-pipeline2', 'children'),
     Output('loading-output-pipeline3', 'children'),     
     Output('post-pipeline-step', 'style'),
     Output('first-step-results', 'children'),  
     Output('second-step-results', 'children'),  
     Output('third-step-results', 'children'),   
     Output('second-step', 'style'),
     Output('third-step', 'style'),
     Output('fourth-step', 'style'),
     Output('first-step-completed', 'data')],
    [Input('run-pipeline', 'n_clicks'),
     Input('run-second-step', 'n_clicks'),
     Input('run-final-step', 'n_clicks')],
    [State('target-column', 'value'),
     State('confounder-columns', 'value'),
     State('subset-length', 'value'),
     State('jump-n', 'value'),
     State('z-score-threshold', 'value'),
     State('resample-freq', 'value'),
     State('embedding-dim', 'value'),
     State('lag', 'value'),
     State('eccm-window-size', 'value'),
     State('stored-filename', 'data'),
     State('stored-output-folder', 'data'),
     State('number-of-cores', 'value'),
     State('ccm-training-proportion', 'value'),
     State('max-mutual-info-shift', 'value'),
     State('num-surrogates-x1', 'value'),
     State('num-surrogates-x2', 'value'),
     State('sig-quant', 'value'),
     State('categorization-mode', 'value'),
     State('categorization-filepath', 'data'),
     State('bn-training-fraction', 'value'),
     State('probability-cutoff', 'value'),
     State('bidirectional-interaction', 'value'),
     State('check-convergence', 'value'),
     State('prefer-zero-lag', 'value'),
     State('known-interactions', 'data'),
     State('first-step-results', 'children'),   
     State('second-step-results', 'children'),  
     State('third-step-results', 'children'),
     State('table-editing', 'data')]
)
def run_pipeline(n_clicks_run, n_clicks_second, n_clicks_final, target_column, confounders, subSetLength, jumpN, z_score_threshold, resample_freq,
                 embedding_dim, lag, eccm_window_size, file_path, output_folder, number_of_cores, ccm_training_prop, max_mi_shift,
                 num_surrogates_x1, num_surrogates_x2, sig_quant, categorization_mode, categorization_filepath,
                 bn_training_fraction, probability_cutoff, bidirectional_interaction, check_convergence, prefer_zero_lag, known_interactions,
                 current_first_results, current_second_results, current_third_results, current_table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Validation checks
    if file_path and os.path.exists(file_path) and subSetLength is not None and eccm_window_size is not None and embedding_dim is not None and lag is not None and resample_freq is not None:
        data_length = len(pd.read_csv(file_path))
        
        if int(subSetLength) > data_length:
            return 'Subset length should be smaller than the input dataset', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False

        if int(eccm_window_size) > data_length:
            return 'ECCM window size should be smaller than the input dataset', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False

        if int(embedding_dim) * int(lag) > data_length:
            return 'Embedding dimension * lag should be smaller than the input dataset', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False
     
        if is_valid_format(str(resample_freq)) == False:
           return 'Resampling frequency should be in the form of 6H, 1D, 10W, 5M, 1Y', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    def present_all_bn_results(output_folder):
        expected_files = [
            "CausalDAG_NET.png",
            "BN_model_confusionMatrix.png",     
            "BN_model_results.png",
            "BN_model_validation.png",
            "sensitivity_barplot.png",
            "CausalDAG_NET_MEAN_MAX.png",  
            "CausalDAG_NET_MEAN_MIN.png",  
        ]
    
        assets_folder = 'assets/'
        image_files = [f for f in expected_files if os.path.exists(os.path.join(output_folder, f))]
    
        for img in image_files:
            shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
    
        timestamp = str(int(datetime.datetime.now().timestamp()))
    
        return html.Div(className='parameter-card', children=[
            html.H3([html.I(className='fas fa-brain', style={'marginRight': '10px'}), 'Bayesian Network Results'], className='card-title'),
            
            html.Div(className='results-gallery', children=[
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Causal Network', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/CausalDAG_NET.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'CausalDAG_NET.png' in image_files else None,
            ]),
            
            html.H4([html.I(className='fas fa-chart-bar', style={'marginRight': '10px'}), 'Model Performance'], 
                    style={'marginTop': '40px', 'marginBottom': '20px', 'color': '#4f46e5'}),
            html.Div(className='results-gallery', children=[
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Model Validation', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/BN_model_validation.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'BN_model_validation.png' in image_files else None,
                
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Prediction Results', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/BN_model_results.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'BN_model_results.png' in image_files else None,
                
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Confusion Matrix', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/BN_model_confusionMatrix.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'BN_model_confusionMatrix.png' in image_files else None,
                
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Sensitivity Analysis', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/sensitivity_barplot.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'sensitivity_barplot.png' in image_files else None,
            ]),
            
            html.H4([html.I(className='fas fa-arrow-up', style={'marginRight': '10px'}), 'High Probability Scenarios'], 
                    style={'marginTop': '40px', 'marginBottom': '20px', 'color': '#4f46e5'}),
            html.P('This diagram shows the average variable values across all high probability scenarios.', 
                   style={'marginBottom': '20px', 'color': '#6b7280', 'fontStyle': 'italic'}),
            html.Div(className='results-gallery', children=[
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Mean High Probability Scenario', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/CausalDAG_NET_MEAN_MAX.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'CausalDAG_NET_MEAN_MAX.png' in image_files else None,
            ]),
            
            html.H4([html.I(className='fas fa-arrow-down', style={'marginRight': '10px'}), 'Low Probability Scenarios'], 
                    style={'marginTop': '40px', 'marginBottom': '20px', 'color': '#4f46e5'}),
            html.P('This diagram shows the average variable values across all low probability scenarios.', 
                   style={'marginBottom': '20px', 'color': '#6b7280', 'fontStyle': 'italic'}),
            html.Div(className='results-gallery', children=[
                html.Div(className='result-card', children=[
                    html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                        html.H4('Mean Low Probability Scenario', style={'margin': '0 0 10px 0', 'color': '#374151'})
                    ]),
                    html.Img(src=f'/assets/CausalDAG_NET_MEAN_MIN.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                ]) if 'CausalDAG_NET_MEAN_MIN.png' in image_files else None,
            ])
        ])

    common_params = {
        "target_column": target_column,
        "confounders": confounders,
        "subSetLength": subSetLength,
        "jumpN": jumpN,
        "z_score_threshold": z_score_threshold,
        "resample_freq": resample_freq,
        "embedding_dim": embedding_dim,
        "lag": lag,
        "eccm_window_size": eccm_window_size,
        "file_path": file_path,
        "number_of_cores": number_of_cores,
        "ccm_training_proportion": ccm_training_prop,
        "max_mi_shift": max_mi_shift,
        "check_convergence": check_convergence,
        "prefer_zero_lag": prefer_zero_lag,
        "known_interactions": known_interactions,
    }

    if button_id == 'run-pipeline':
        if not target_column or not file_path:
            return 'Please select a target column and upload a file.', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False

        step_name = "CCM ECCM Step"
        write_parameters_to_file(output_folder, step_name, common_params)

        command = [
            'python', '1_CCM_ECCM.py',
            '--output_folder', output_folder+"/",
            '--target_column', target_column,
            '--confounders', ','.join(confounders) if confounders else "",
            '--subSetLength', str(subSetLength),
            '--jumpN', str(jumpN),
            '--z_score_threshold', str(z_score_threshold),
            '--resample_freq', resample_freq,
            '--embedding_dim', str(embedding_dim),
            '--lag', str(lag),
            '--eccm_window_size', str(eccm_window_size),
            '--file_path', file_path,
            '--number_of_cores', str(number_of_cores),
            '--ccm_training_proportion', str(ccm_training_prop),
            '--max_mi_shift', str(max_mi_shift),
            '--check_convergence', str(check_convergence),
            '--prefer_zero_lag', str(prefer_zero_lag)
        ]
        
        try:
            clear_assets_folder()
            for file in os.listdir(output_folder):
                if file.startswith('ccm_density_') and file.endswith('.png'):
                    os.remove(os.path.join(output_folder, file))
                elif file.startswith('eccm_') and file.endswith('.png'):
                    os.remove(os.path.join(output_folder, file))
                    
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=360000)

            if process.returncode != 0:
                return 'Error in CCM ECCM step execution: ' + stderr.decode(), '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False

            known_status = incorporate_known_interactions(output_folder, known_interactions)

            try:
                curated_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
                if os.path.exists(curated_file):
                    df_results_with_known = pd.read_csv(curated_file)
                    visualize_network(df_results_with_known, os.path.join(output_folder, 'network_plot.png'))
            except Exception as e:
                print(f"Error regenerating network plot with known interactions: {e}")

            shutil.copy(os.path.join(output_folder, 'network_plot.png'), os.path.join('assets', 'network_plot.png'))
            
            timestamp = str(int(datetime.datetime.now().timestamp()))

            success_message = 'CCM ECCM step completed successfully.'
            if known_interactions:
                success_message += f' Known interactions: {known_status}'

            # Load actual CSV data for image controls
            actual_table_data = None
            try:
                curated_file = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
                if os.path.exists(curated_file):
                    df_csv = pd.read_csv(curated_file)
                    # Convert is_Valid to integers
                    if 'is_Valid' in df_csv.columns:
                        df_csv['is_Valid'] = pd.to_numeric(df_csv['is_Valid'], errors='coerce').fillna(2).astype(int)
                    actual_table_data = df_csv.to_dict('records')
                    with open('debug_log.txt', 'a') as f:
                        f.write(f"MAIN PIPELINE: Loaded {len(actual_table_data)} rows from CSV for image controls\n")
                        if actual_table_data:
                            invalid_count = sum(1 for row in actual_table_data if row.get('is_Valid') == 0)
                            f.write(f"MAIN PIPELINE: Found {invalid_count} invalid entries\n")
            except Exception as e:
                with open('debug_log.txt', 'a') as f:
                    f.write(f"MAIN PIPELINE: Error loading CSV for images: {e}\n")
            
            # Use the new functions with controls and actual CSV data
            first_step_results = [html.Div(className='progress-section', children=[
                html.Div(className='step-indicator', children=[
                    html.Div('1', className='step-number'),
                    html.H2('Initial Analysis Complete', className='step-title')
                ]),
                present_all_density_ccm_with_controls(output_folder, actual_table_data),
                present_all_eccm_with_controls(output_folder, actual_table_data),
                html.Div(className='parameter-card', children=[
                    html.H3([html.I(className='fas fa-network-wired', style={'marginRight': '10px'}), 'Network Visualization'], className='card-title'),
                    html.Div(style={'textAlign': 'center'}, children=[
                        html.Img(src=f'assets/network_plot.png?t={timestamp}', style={'maxWidth': '100%', 'height': 'auto', 'borderRadius': '10px'})
                    ])
                ])
            ])]

            return (success_message, '', '', '', {'display': 'block'}, 
                    first_step_results, current_second_results, current_third_results,  
                    {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True)
                                
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait()
            except:
                pass
            return 'CCM ECCM process timed out after 1 hour.', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False
        except Exception as e:
            return f'Error running CCM ECCM: {str(e)}', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False

    elif button_id == 'run-second-step':
        step_name = "Surrogates Step"
        step_params = {
            "num_surrogates_x1": num_surrogates_x1,
            "num_surrogates_x2": num_surrogates_x2,
            "sig_quant": sig_quant,
        }
        write_parameters_to_file(output_folder, step_name, {**common_params, **step_params})

        # Check if manual edits were made
        manual_edits_flag = os.path.join(output_folder, 'MANUAL_EDITS_MADE.flag')
        preserve_edits = os.path.exists(manual_edits_flag)

    
        command = [
            'python', '2_SURR.py',
            '--file_path', file_path,
            '--output_folder', output_folder, 
            '--preserve_manual_edits', str(preserve_edits),  # ADD THIS LINE
            '--subSetLength', str(subSetLength),
            '--jumpN', str(jumpN),
            '--z_score_threshold', str(z_score_threshold),
            '--resample_freq', resample_freq,
            '--embedding_dim', str(embedding_dim),
            '--lag', str(lag),
            '--num_surrogates_x1', str(num_surrogates_x1),
            '--num_surrogates_x2', str(num_surrogates_x2),
            '--sig_quant', str(sig_quant),
            '--number_of_cores', str(number_of_cores),
            '--ccm_training_proportion', str(ccm_training_prop),
            '--max_mi_shift', str(max_mi_shift),
        ]
        
        try:
            surrogate_files = [f for f in os.listdir('assets') if 'surr' in f.lower()]
            for f in surrogate_files:
                try:
                    os.remove(os.path.join('assets', f))
                except:
                    pass
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()  # No timeout - can take days
    
            if process.returncode != 0:
                error_msg = f'Error in Surrogates step execution (return code: {process.returncode}): {stderr.decode()}'
                return error_msg, '', error_msg, '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True
    
            surr_plot_source = os.path.join(output_folder, 'Surr_plot.png')
            surr_plot_dest = os.path.join('assets', 'Surr_plot.png')
            surrogate_plot_available = False
            
            if os.path.exists(surr_plot_source):
                try:
                    os.makedirs('assets', exist_ok=True)
                    shutil.copy(surr_plot_source, surr_plot_dest)
                    surrogate_plot_available = True
                except Exception as e:
                    print(f"Error copying Surr_plot.png: {e}")
            else:
                plot_files = [f for f in os.listdir(output_folder) if 'surr' in f.lower() and f.endswith('.png')]
                if plot_files:
                    fallback_file = plot_files[0]
                    fallback_source = os.path.join(output_folder, fallback_file)
                    try:
                        shutil.copy(fallback_source, surr_plot_dest)
                        surrogate_plot_available = True
                    except Exception as e:
                        print(f"Error copying fallback file {fallback_file}: {e}")
    
            network_plot_source = os.path.join(output_folder, 'network_plot.png')
            network_plot_dest = os.path.join('assets', 'network_plot.png')
            network_plot_available = False
            
            if os.path.exists(network_plot_source):
                try:
                    shutil.copy(network_plot_source, network_plot_dest)
                    network_plot_available = True
                except Exception as e:
                    print(f"Error copying network plot: {e}")
            
            timestamp = str(int(datetime.datetime.now().timestamp()))
            
            surrogate_cards = []
            
            if network_plot_available:
                surrogate_cards.append(
                    html.Div(className='result-card', children=[
                        html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                            html.H4('Updated Network Visualization', style={'margin': '0 0 10px 0', 'color': '#374151'})
                        ]),
                        html.Img(src=f'assets/network_plot.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                    ])
                )
            
            if surrogate_plot_available:
                surrogate_cards.append(
                    html.Div(className='result-card', children=[
                        html.Div(style={'padding': '15px', 'background': '#f8fafc'}, children=[
                            html.H4('Surrogate Analysis Results', style={'margin': '0 0 10px 0', 'color': '#374151'})
                        ]),
                        html.Img(src=f'/assets/Surr_plot.png?t={timestamp}', style={'width': '100%', 'height': 'auto'})
                    ])
                )
            else:
                surrogate_cards.append(
                    html.Div(className='result-card', children=[
                        html.Div(style={'padding': '15px', 'background': '#fff3cd', 'border': '1px solid #ffeaa7'}, children=[
                            html.H4('Surrogate Plot Not Available', style={'margin': '0 0 10px 0', 'color': '#856404'}),
                            html.P('The surrogate analysis completed but did not generate the expected plot.', 
                                   style={'margin': '0', 'color': '#856404', 'fontSize': '0.9rem'})
                        ])
                    ])
                )
            
            second_step_results = [html.Div(className='progress-section', children=[
                html.Div(className='step-indicator', children=[
                    html.Div('3', className='step-number'),
                    html.H2('Surrogate Analysis Complete', className='step-title')
                ]),
                html.Div(className='parameter-card', children=[
                    html.H3([html.I(className='fas fa-chart-bar', style={'marginRight': '10px'}), 'Surrogate Analysis Results'], className='card-title'),
                    html.Div(className='results-gallery', children=surrogate_cards)
                ])
            ])]
            
            success_msg = 'Surrogates step completed successfully.'
            if not surrogate_plot_available:
                success_msg += ' (Note: Surrogate plot was not generated)'
            
            return (success_msg, '', '', '', {'display': 'none'}, 
                    current_first_results, second_step_results, current_third_results,  
                    {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True)
                
        except Exception as e:
            import traceback
            error_msg = f'Error running Surrogates: {str(e)}\n{traceback.format_exc()}'
            return error_msg, '', error_msg, '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True

    elif button_id == 'run-final-step':
        categorization = categorization_filepath if categorization_mode == 'upload' else 'auto'
        
        if categorization_mode == 'upload' and not categorization_filepath:
            return 'Please upload a categories file.', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True
        
        number_of_random_vecs = 0
        
        step_name = "BN Step"
        step_params = {
            "categorization_mode": categorization_mode,
            "categorization_filepath": categorization_filepath,
            "bn_training_fraction": bn_training_fraction,
            "number_of_random_vecs": number_of_random_vecs,
            "probability_cutoff": probability_cutoff,
            "bidirectional_interaction": bidirectional_interaction,
        }
        write_parameters_to_file(output_folder, step_name, {**common_params, **step_params})

        command = [
            'python', '3_BN.py',
            '--file_path', file_path,
            '--output_folder', output_folder+"/",            
            '--target_column', target_column,
            '--confounders', ','.join(confounders) if confounders else "",
            '--subSetLength', str(subSetLength),
            '--jumpN', str(jumpN),
            '--z_score_threshold', str(z_score_threshold),
            '--resample_freq', resample_freq,
            '--embedding_dim', str(embedding_dim),
            '--lag', str(lag),
            '--number_of_cores', str(number_of_cores),
            '--ccm_training_proportion', str(ccm_training_prop),
            '--max_mi_shift', str(max_mi_shift),
            '--auto_categorization', 'auto' if categorization_mode == 'auto' else '',
            '--categorization', categorization if categorization_mode == 'upload' else '',
            '--bn_training_fraction', str(bn_training_fraction),
            '--number_of_random_vecs', str(number_of_random_vecs),
            '--probability_cutoff', str(probability_cutoff),
            '--bidirectional_interaction', bidirectional_interaction,
        ]
        
        try:
            important_plots = ['Surr_plot.png', 'network_plot.png']
            temp_plots = {}
            for plot in important_plots:
                plot_path = os.path.join('assets', plot)
                if os.path.exists(plot_path):
                    with open(plot_path, 'rb') as f:
                        temp_plots[plot] = f.read()
            
            clear_assets_folder()
            
            for plot, data in temp_plots.items():
                with open(os.path.join('assets', plot), 'wb') as f:
                    f.write(data)
            
            # Copy CSV to tmp folder for stable BN execution
            tmp_folder = os.path.join(output_folder, 'tmp')
            os.makedirs(tmp_folder, exist_ok=True)
            original_csv = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
            tmp_csv = os.path.join(tmp_folder, 'CCM_ECCM_curated.csv')
            if os.path.exists(original_csv):
                shutil.copy2(original_csv, tmp_csv)
            
            with open('debug_log.txt', 'a') as f:
                f.write(f"\n=== BN SUBPROCESS START ===\n")
                f.write(f"Time: {datetime.datetime.now()}\n")
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"CSV copied to tmp\n")
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=300000)  # 5 minutes - reasonable for BN calculations

            with open('debug_log.txt', 'a') as f:
                f.write(f"\n=== BN SUBPROCESS COMPLETE ===\n")
                f.write(f"Time: {datetime.datetime.now()}\n")
                f.write(f"Return code: {process.returncode}\n")
                f.write(f"Stdout: {stdout.decode()[:1000]}\n")
                f.write(f"Stderr: {stderr.decode()[:1000]}\n")

            if process.returncode != 0:
                return 'Error in BN step execution: ' + stderr.decode(), '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True
            
            important_plots = ['Surr_plot.png', 'network_plot.png']
            for plot in important_plots:
                source_path = os.path.join(output_folder, plot)
                dest_path = os.path.join('assets', plot)
                if os.path.exists(source_path):
                    try:
                        shutil.copy(source_path, dest_path)
                    except:
                        pass
            
            # Restore CSV FIRST to preserve user changes
            # Restore CSV BEFORE creating results to preserve user changes
            manual_edits_flag = os.path.join(output_folder, 'MANUAL_EDITS_MADE.flag')
            try:
                if os.path.exists(tmp_csv):
                    # Debug: check tmp CSV content before restore
                    tmp_df = pd.read_csv(tmp_csv)
                    with open('debug_log.txt', 'a') as f:
                        f.write(f"\n=== CSV RESTORE DEBUG ===\n")
                        f.write(f"TMP CSV has {len(tmp_df)} rows\n")
                        f.write(f"TMP is_Valid sample: {tmp_df['is_Valid'].head(5).tolist()}\n")
                    
                    shutil.copy2(tmp_csv, original_csv)
                    
                    # Debug: verify restore worked
                    restored_df = pd.read_csv(original_csv)
                    with open('debug_log.txt', 'a') as f:
                        f.write(f"RESTORED CSV has {len(restored_df)} rows\n")
                        f.write(f"RESTORED is_Valid sample: {restored_df['is_Valid'].head(5).tolist()}\n")
                    
                    # Create manual edits flag to prevent table reset
                    with open(manual_edits_flag, 'w') as f:
                        f.write('Manual edits preserved after BN step')
                    shutil.rmtree(tmp_folder)
            except Exception as e:
                with open('debug_log.txt', 'a') as f:
                    f.write(f"CSV restore failed: {e}\n")
            
            # Ensure manual edits flag exists regardless
            if not os.path.exists(manual_edits_flag):
                with open(manual_edits_flag, 'w') as f:
                    f.write('Manual edits preserved after BN step')
            
            # NOW create results after CSV is restored
            third_step_results = [html.Div(className='progress-section', children=[
                html.Div(className='step-indicator', children=[
                    html.Div('5', className='step-number'),
                    html.H2('Bayesian Network Analysis Complete', className='step-title')
                ]),
                present_all_bn_results(output_folder)  
            ])]
            
            return ('BN step completed successfully.', '', '', '', {'display': 'none'}, 
                    current_first_results, current_second_results, third_step_results,   
                    {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, True)
                
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait()
            except:
                pass
            return 'BN process timed out after 100 hours.', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True
        except Exception as e:
            return f'Error running BN: {str(e)}', '', '', '', {'display': 'none'}, current_first_results, current_second_results, current_third_results, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, True

    raise PreventUpdate

# Clientside callbacks for loading and UI management
app.clientside_callback(
    """
    function(n_clicks_run, n_clicks_second, n_clicks_final) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) {
            return {'display': 'none'};
        }
        
        const button_id = ctx.triggered[0].prop_id.split('.')[0];
        if (button_id === 'run-pipeline' || button_id === 'run-second-step' || button_id === 'run-final-step') {
            return {'display': 'block'};
        }
        return {'display': 'none'};
    }
    """,
    Output('persistent-loading', 'style'),
    [Input('run-pipeline', 'n_clicks'),
     Input('run-second-step', 'n_clicks'), 
     Input('run-final-step', 'n_clicks')]
)

app.clientside_callback(
    """
    function(pipeline_output) {
        if (pipeline_output && (typeof pipeline_output === 'string' || pipeline_output.length > 0)) {
            return {'display': 'none'};
        }
        return dash_clientside.no_update;
    }
    """,
    Output('persistent-loading', 'style', allow_duplicate=True),
    [Input('pipeline-output', 'children')],
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks_run, n_clicks_second, n_clicks_final) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) {
            return ['Processing...', 'Please wait...'];
        }
        
        const button_id = ctx.triggered[0].prop_id.split('.')[0];
        
        if (button_id === 'run-pipeline') {
            return ['Running CCM-ECCM Analysis...', 'This process may take few minutes to several hours depending on your data size and parameters. Please do not close this window.'];
        } else if (button_id === 'run-second-step') {
            return ['Running Surrogate Analysis...', 'Performing statistical validation of causal relationships. This may take few minutes to several hours.'];
        } else if (button_id === 'run-final-step') {
            return ['Building Bayesian Network...', 'Constructing probabilistic model and generating predictions. This may take few minutes.'];
        }
        
        return ['Processing...', 'Please wait...'];
    }
    """,
    [Output('persistent-message', 'children'),
     Output('persistent-details', 'children')],
    [Input('run-pipeline', 'n_clicks'),
     Input('run-second-step', 'n_clicks'),
     Input('run-final-step', 'n_clicks')]
)

if __name__ == '__main__':
    app.run(debug=False, port=8050, host='127.0.0.1')