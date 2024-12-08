import dash
from dash import dcc, html, dash_table, Input, Output, State
import base64
import io
import pandas as pd
import subprocess
from dash.exceptions import PreventUpdate
import tempfile
import os, re
import shutil
import datetime
import networkx as nx
import matplotlib.pyplot as plt


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
    
    for filename in os.listdir(assets_folder):
        file_path = os.path.join(assets_folder, filename)
        
        # Skip the 'logo' folder
        if file_path == logo_folder:
            continue
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app._favicon = ("logo/logo.png")
app.title = 'CEcBaN'
clear_assets_folder()  

def write_parameters_to_file(output_folder, step_name, params):
    with open(os.path.join(output_folder, 'parameters.txt'), 'a') as file:
        file.write(f"Step: {step_name}\n")
        for param, value in params.items():
            file.write(f"{param}: {value}\n")
        file.write("\n")

instructions_text = html.H3("The CEcBaN tool is a user-friendly implementation of the causal approach described in Tal, et al., 2024. Briefly, the approach aims: (1) to identify the causal interactions within a complex dynamic system described by a multivariate time series. (2) to reconstruct the structure of the causal interactions which eventually affect a certain variable (target). (3) Supply insights regarding different environmental scenarios.\
                             CEcBaN takes as input a multivariate time series and generates the following outputs: (1) CCM results for all possible interactions between the input variables and a selected target variable, calculated based on user-selected parameters; (2) CCM results for all possible interactions between the causal variables identified in step 1; (3) ECCM results for the suggested causal interactions from steps 1 and 2; (4) statistical analysis using comparison to \
                             surrogate results; (5) a DAG of the causal network, constructed based on user-selected conditions; and (6) a Bayesian network model, including mean scenarios that describe the dynamics leading to higher or lower target values. \
                             ")        
                             
app.layout = html.Div([
    dcc.Store(id='stored-output-folder', storage_type='session'),  
    html.Div(
    [
        html.Div(
            [
                html.Img(src='/assets/logo/logo.png', style={'width': '10%', 'height': 'auto'}),
                html.H1(
                    'CEcBaN Tool',
                    style={'textAlign': 'center', 'flex': '1'}
                ),
                html.Img(src='/assets/logo/logo_iolr.png', style={'width': '10%', 'height': 'auto'})
            ],
            style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}
        )
    ]
),
    html.H2("Tal Lab", style={'textAlign': 'center', 'flex': '1'}),
    html.H3("CEcBaN - CCM ECCM Bayesian Network", style={'textAlign': 'center', 'flex': '1'}),
    instructions_text,
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload', style={'marginBottom': '20px'}),
    dcc.Store(id='stored-filename'),  
    dcc.Store(id='stored-columns'), 
    dcc.Store(id='categorization-filepath'),  
    dcc.Store(id='first-step-completed', data=False), 
    html.Div([
        html.Label("Select Target Column: [Can be only a single target in the analysis. This is the variable that is set as the target variable. It lacks of out-edges, and is the last node of the DAG", style={'marginBottom': '10px'}),
        dcc.Dropdown(id='target-column'),
    ], style={'marginBottom': '20px',}),
    html.Div([
        html.Label("Select Confounders: [Can be none, single or multiple confounders. The confounders only affect the system, and not affected by any of the other variables. A confounder consists only out-edges.]", style={'marginBottom': '10px'}),
        dcc.Checklist(id='confounder-columns')
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Enter SubSet Length: [This is the size of the sliding window.]"),
        dcc.Input(id='subset-length', type='number', value=60),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Enter Jump N: [The number of timesteps between the beginning of window n to the beginning of window n+1.]"),
        dcc.Input(id='jump-n', type='number', value=30),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Z-score Threshold for Outliers: [values above this Z-score will be discarded, and replaced by an interpolated value.]"),
        dcc.Input(id='z-score-threshold', type='number', value=3),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Resampling Frequency (e.g., ‘6H’, ‘5D’, ‘1W’, ‘1M’, ‘1Y’): [resampling of the timeseries according to the format of <int><H/D/W/M/Y>.]"),
        dcc.Input(id='resample-freq', type='text', value='5D'),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Embedding Dimension (E): [The maximal embedding dimention that the simplex projection analysis is allowed. if higher, default E = 5 is set. If E < 3, default E = 3 is set.]"),
        dcc.Input(id='embedding-dim', type='number', value=2),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Lag (l): [This is the lag used for the embeddings calculation alongside the selected E.]"),
        dcc.Input(id='lag', type='number', value=1),
    ], style={'marginBottom': '20px'}),
    
    html.Div([
        html.Label("For convergence measurement:"),
        dcc.Dropdown(
            id='check-convergence',
            options=[
                {'label': 'Use results mean to check convergence', 'value': 'means'},
                {'label': 'Use results density to check convergence ', 'value': 'density'}
            ],
            value='density'
        )
    ], style={'marginBottom': '20px', 'width': '20%'}),
    
    html.Div([
        html.Label("ECCM Window Size: [The size of the window that will be used in the ECCM calculations step.]"),
        dcc.Input(id='eccm-window-size', type='number', value=50),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Number of Cores:[The pipeline is highly parallelized. This is the number of cores to be used for multithreading.]"),
        dcc.Input(id='number-of-cores', type='number', value=1),  
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("CCM Training Proportion [The fruction of train (train:test split)]:"),
        dcc.Input(id='ccm-training-proportion', type='number', value=0.75, min=0, max=1, step=0.01),
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.Label("Max Mutual Information Shift: [The number of maximal timesteps shift in the ECCM calculation]"),
        dcc.Input(id='max-mutual-info-shift', type='number', value=20),
    ], style={'marginBottom': '20px'}),
    html.Div([
        dcc.Loading(
            id="loading-pipeline",
            type="circle",
            children=html.Div(id="loading-output-pipeline")
        )
    ], style={'marginBottom': '20px'}),
    html.Div(id='post-pipeline-step', style={'display': 'none'}),  
    html.Button('Run CCM ECCM', id='run-pipeline', n_clicks=0, style={'marginTop': '20px'}),
    html.Div(id='pipeline-output'),
    html.Div(id='image-1'),  
    html.Div(id='second-step', children=[
        html.H6("Editable interactions network file. Please review the CCM and ECCM results and refine the network structure accordingly. the \"2\" under is_Valid column indicates that the interaction is approved."),
        dash_table.DataTable(
            id='table-editing',
            columns=[],
            data=[],
            editable=True
        ),
        html.Button('Save Edits', id='save-edits', n_clicks=0, style={'marginTop': '20px'}),
        html.Div(id='save-confirmation', style={'marginTop': '20px'}),
        
        html.Label("Number of Surrogates for x1:"),
        dcc.Input(id='num-surrogates-x1', type='number', value=33),
        html.Label("Number of Surrogates for x2:"),
        dcc.Input(id='num-surrogates-x2', type='number', value=33),
        html.Label("Significance Quantile:"),
        dcc.Input(id='sig-quant', type='number', value=0.95),
        html.Div(),
        html.Div([
            dcc.Loading(
                id="loading-pipeline2",
                type="circle",
                children=html.Div(id="loading-output-pipeline2")
            )
        ], style={'marginBottom': '20px'}),
        html.Button('Run calculations using surrogate dataset', id='run-second-step', n_clicks=0, style={'marginTop': '20px'}),
        html.Button('Edit Interactions Network', id='edit-interactions-network', n_clicks=0, style={'marginTop': '20px', 'display': 'none'})
    ], style={'display': 'none'}),
    html.Div(id='third-step', children=[
        html.Div([
            html.Label("Categorization: [The tool categorizes the data for the BN model calculations. \
                       The automatic categorization scans all of the possible categories (using min gap between quantiles = 0.1). \
                        Next, the Mann Whitney test is used to calculate whether the different groups are significantly distinguished. \
                        If multiple vectors are of similar number of significantly distinguished groups, the one which is divided most evenly is selected. \
                        If Upload Categories File is selected, the categories are determined by the user for each variable. See docs folder for an example file.]"),
            dcc.RadioItems(
                id='categorization-mode',
                options=[
                    {'label': 'Automatic', 'value': 'auto'},
                    {'label': 'Upload Categories File', 'value': 'upload'}
                ],
                value='auto',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'marginBottom': '20px'}),
        
        html.Div(id='upload-categorization-file', children=[
            html.Label("Upload Categories File:"),
            dcc.Upload(
                id='upload-categories-file',
                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-categories-upload', style={'marginBottom': '20px'})
        ], style={'display': 'none'}),

        html.Div([
            html.Label("Training Fraction: [This is the training fruction used in the BN model training process.]"),
            dcc.Input(id='bn-training-fraction', type='number', value=0.75, min=0, max=1, step=0.01)
        ], style={'marginBottom': '20px'}),
        #html.Div([
        #    html.Label("Number of Random Vectors: [Each random vector is a scenario testing. The tool usees many random environmental scenarios to find the mean Max and mean Min scenarios. More vectors would scan more different scenarios.]"),
        #    dcc.Input(id='number-of-random-vecs', type='number', value=100)
        #], style={'marginBottom': '20px'}),
        html.Div([
            html.Label("Probability Cutoff: [This is the cutoff that the tool uses to interprete models output.]"),
            dcc.Input(id='probability-cutoff', type='number', value=0.5)
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Label("When bidirectional interaction, keep: [To solve conflicts in DAG construction, this rule determines which of the interactions will be used.]"),
            dcc.Dropdown(
                id='bidirectional-interaction',
                options=[
                    {'label': 'Keep higher CCM score', 'value': 'higher'},
                    {'label': 'Keep earlier effect', 'value': 'earlier'}
                ],
                value='higher'
            )
        ], style={'marginBottom': '20px', 'width': '20%'}),
        
        html.Div([
            dcc.Loading(
                id="loading-pipeline3",
                type="circle",
                children=html.Div(id="loading-output-pipeline3")
            )
        ], style={'marginBottom': '20px'}),
        html.Button('Build and evaluate BN model', id='run-final-step', n_clicks=0, style={'marginTop': '20px'})
    ], style={'display': 'none'}),
])

@app.callback(
    Output('stored-output-folder', 'data'),
    Input('upload-data', 'children'),  # Triggered when the page loads
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
                    html.H5(filename),
                    html.H6("Preview of the first 10 rows:"),
                    dash_table.DataTable(
                        data=df.head(10).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df.columns]
                    )
                ]),
                columns_options,
                columns_options,
                temp_file.name,
                columns
            )
        except Exception as e:
            return (
                html.Div([
                    'There was an error processing this file: ' + str(e)
                ]),
                [], [], None, []
            )
    return (html.Div('Upload CSV file to see a preview.'), [], [], None, [])

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

# Save the edited table data when "Save Edits" is clicked
@app.callback(
    Output('save-confirmation', 'children'),
    [Input('save-edits', 'n_clicks')],
    [State('table-editing', 'data'),
     State('table-editing', 'columns'),
     State('stored-output-folder', 'data')]  
)
def save_edited_table(n_clicks, rows, columns, output_folder):
    if n_clicks > 0 and output_folder:
        try:
            df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
            
            d = datetime.datetime.today()
           
            output_filepath = os.path.join(output_folder, 'CCM_ECCM_curated.csv')
            
            output_filepath_archived = os.path.join(os.getcwd(), output_folder, "CCM_ECCM_curated_"+ f"{d.year}{d.month}{d.day}{d.hour}{d.minute}{d.second}"+".csv")

            df_previous = pd.read_csv(output_filepath)
            df_previous.to_csv(output_filepath_archived, index=False)
            
            df.to_csv(output_filepath, index=False)
           
            df_results = pd.read_csv(output_filepath)

            #df_results = df_results[df_results["Score"] > 0]
            visualize_network(df_results, os.path.join(output_folder, 'network_plot.png'))
            shutil.copy(os.path.join(output_folder, 'network_plot.png'), os.path.join('assets', 'network_plot.png'))        
            
            return html.Div("Edits have been saved successfully!", style={'color': 'green', 'marginTop': '20px'})
        except Exception as e:
            return html.Div(f"Failed to save edits: {str(e)}", style={'color': 'red', 'marginTop': '20px'})
    
    return ''

@app.callback(
    [Output('table-editing', 'columns'),
     Output('table-editing', 'data'),
     Output('third-step', 'style')],
    [Input('first-step-completed', 'data')],
    [State('stored-output-folder', 'data')]
)
def load_curated_file(first_step_completed, output_folder):
    if first_step_completed:
        try:
            # Check if the file exists and load it
            df = pd.read_csv(os.path.join(output_folder, 'CCM_ECCM_curated.csv'))
            columns = [{"name": i, "id": i} for i in df.columns]
            data = df.to_dict('records')
            return columns, data, {'display': 'block'}
        except FileNotFoundError:
            # If the file is not yet created, wait until it becomes available
            return [], [], {'display': 'none'}
    return [], [], {'display': 'none'}

@app.callback(
    [Output('pipeline-output', 'children'),
     Output('loading-output-pipeline', 'children'),
     Output('loading-output-pipeline2', 'children'),
     Output('loading-output-pipeline3', 'children'),     
     Output('post-pipeline-step', 'style'),
     Output('image-1', 'children'),
     Output('second-step', 'style'),
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
     #State('number-of-random-vecs', 'value'),
     State('probability-cutoff', 'value'),
     State('bidirectional-interaction', 'value'),
     State('check-convergence', 'value')]
)
def run_pipeline(n_clicks_run, n_clicks_second, n_clicks_final, target_column, confounders, subSetLength, jumpN, z_score_threshold, resample_freq,
                 embedding_dim, lag, eccm_window_size, file_path, output_folder, number_of_cores, ccm_training_prop, max_mi_shift,
                 num_surrogates_x1, num_surrogates_x2, sig_quant, categorization_mode, categorization_filepath,
                 bn_training_fraction, probability_cutoff, bidirectional_interaction, check_convergence):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    if int(subSetLength) > len(pd.read_csv(file_path)):
        return 'Subset length should be smaller than the input dataset', '', '', '', {'display': 'none'}, None, {'display': 'none'}, False

    if int(eccm_window_size) > len(pd.read_csv(file_path)):
        return 'ECCM window size should be smaller than the input dataset', '', '', '', {'display': 'none'}, None, {'display': 'none'}, False

    if int(embedding_dim) * int(lag) > len(pd.read_csv(file_path)):
        return 'Embedding dimension * lag should be smaller than the input dataset', '', '', '', {'display': 'none'}, None, {'display': 'none'}, False
     
    if is_valid_format(str(resample_freq)) == False:
       return 'Resampling frequency should be in the form of 6H, 1D, 10W, 5M, 1Y', '', '', '', {'display': 'none'}, None, {'display': 'none'}, False
   
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    def present_all_density_ccm(output_folder):
        prefix = "ccm_density_"
        assets_folder = 'assets/'
        
        image_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(".png")]
        
        # Copy images to assets folder
        for img in image_files:
            shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
        
        return html.Div([
            html.H1("CCM Results"),
            html.Div([
                html.Img(src=f'/assets/{img}', style={'width': '25%', 'display': 'inline-block'}) for img in image_files
            ])
        ])
    
    def present_all_eccm(output_folder):
        prefix = "eccm_"
        assets_folder = 'assets/'
        
        image_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(".png")]
        
        # Copy images to assets folder
        for img in image_files:
            shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
        
        return html.Div([
            html.H1("ECCM Results"),
            html.Div([
                html.Img(src=f'/assets/{img}', style={'width': '25%', 'display': 'inline-block'}) for img in image_files
            ])
        ])
    
    
    def present_all_bn_results(output_folder):
        expected_files = [
            "ccm_eccm.png",
            "ccm_dag.png",        
            "CausalDAG_NET.png",
            "BN_model_confusionMatrix.png",     
            "BN_model_results.png",
            "BN_model_validation.png",
            "sensitivity_barplot.png",
            "CausalDAG_NET_MAX.png",
            "CausalDAG_NET_MIN.png",
        ]
    
        assets_folder = 'assets/'
        image_files = [f for f in expected_files if os.path.exists(os.path.join(output_folder, f))]
    
        for img in image_files:
            shutil.copy(os.path.join(output_folder, img), os.path.join(assets_folder, img))
    
        return html.Div([
                        html.H1("Bayesian Network Results"),
                        
                        #  1: ccm_eccm and ccm_dag
                        html.H2("Cross-Correlation Maps and Directed Acyclic Graph"),
                        html.Div([
                            html.Img(src=f'/assets/ccm_eccm.png', style={'width': '48%', 'display': 'inline-block', 'margin-right': '2%'}),
                            html.Img(src=f'/assets/ccm_dag.png', style={'width': '48%', 'display': 'inline-block'}),
                        ], style={'text-align': 'center', 'margin-bottom': '20px'}),
                        
                        #  2: CausalDAG_NET
                        html.H2("Causal DAG Network"),
                        html.Div([
                            html.Img(src=f'/assets/CausalDAG_NET.png', style={'width': '30%', 'display': 'inline-block'}),
                        ], style={'text-align': 'center', 'margin-bottom': '20px'}),
                        
                        #  3: BN Model Validation, Results, and Confusion Matrix
                        html.H2("BN Model Evaluation"),
                        html.Div([
                            html.Img(src=f'/assets/BN_model_validation.png', style={'width': '31%', 'display': 'inline-block', 'margin-right': '2%'}),
                            html.Img(src=f'/assets/BN_model_results.png', style={'width': '31%', 'display': 'inline-block', 'margin-right': '2%'}),
                            html.Img(src=f'/assets/BN_model_confusionMatrix.png', style={'width': '31%', 'display': 'inline-block'}),
                        ], style={'text-align': 'center', 'margin-bottom': '20px'}),
                        
                        #  4: CausalDAG_NET_MAX and CausalDAG_NET_MIN
                        html.Div([
                            html.H2("Mean Extreme Scenarios"),
                            html.Div([
                                html.Div([
                                    html.H3("Mean Max Scenario", style={'text-align': 'center'}),
                                    html.Img(src=f'/assets/CausalDAG_NET_MAX.png', style={'width': '30%', 'display': 'inline-block', 'margin-right': '2%'}),
                                ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '48%'}),
                                
                                html.Div([
                                    html.H3("Mean Min Scenario", style={'text-align': 'center'}),
                                    html.Img(src=f'/assets/CausalDAG_NET_MIN.png', style={'width': '30%', 'display': 'inline-block'}),
                                ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '48%'}),
                            ], style={'text-align': 'center', 'margin-bottom': '20px'}),
                        ]),
                        
                        #  5: Sensitivity Analysis
                        html.H2("Sensitivity Analysis"),
                        html.Div([
                            html.Img(src=f'/assets/sensitivity_barplot.png', style={'width': '30%', 'display': 'inline-block'}),
                        ], style={'text-align': 'center'}),
                    ])

                

    # Prepare  parameters
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
    }

    if button_id == 'run-pipeline':
        if not target_column or not file_path:
            return 'Please select a target column and upload a file.', '', '', '', {'display': 'none'}, None, {'display': 'none'}, False

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
            '--check_convergence', str(check_convergence)
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return 'Error in CCM ECCM step execution: ' + stderr.decode(), '', '', '', {'display': 'none'}, None, {'display': 'none'}, False

        shutil.copy(os.path.join(output_folder, 'network_plot.png'), os.path.join('assets', 'network_plot.png'))

        return ('CCM ECCM step completed successfully.', '', '', '', {'display': 'block'},
                [#instructions_text,
                 present_all_density_ccm(output_folder),
                 present_all_eccm(output_folder),
                 html.Img(src='assets/network_plot.png')],
                {'display': 'block'}, True)

    elif button_id == 'run-second-step':
        step_name = "Surrogates Step"
        step_params = {
            "num_surrogates_x1": num_surrogates_x1,
            "num_surrogates_x2": num_surrogates_x2,
            "sig_quant": sig_quant,
        }
        write_parameters_to_file(output_folder, step_name, {**common_params, **step_params})

        command = [
            'python', '2_SURR.py',
            '--file_path', file_path,
            '--output_folder', output_folder+"/",
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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    
        if process.returncode != 0:
            return 'Error in Surrogates step execution: ' + stderr.decode(), '', '', '', {'display': 'none'}, None, {'display': 'none'}, False


        clear_assets_folder()
        try:
            shutil.copy(os.path.join(output_folder, 'Surr_plot.png'), os.path.join('assets', 'Surr_plot.png'))
        except:
            print()
            
        shutil.copy(os.path.join(output_folder, 'network_plot.png'), os.path.join('assets', 'network_plot.png'))
        
        return 'Surrogates step completed successfully.', '', '', '', {'display': 'none'}, [
           # instructions_text,
            present_all_density_ccm(output_folder),
            present_all_eccm(output_folder),
            html.H2("Surrogate Datasets CCM Results"),
            html.Img(src='assets/network_plot.png'),
            html.Img(src='assets/Surr_plot.png')], {'display': 'block'}, True

    elif button_id == 'run-final-step':
        categorization = categorization_filepath if categorization_mode == 'upload' else 'auto'
        
        if categorization_mode == 'upload' and not categorization_filepath:
            return 'Please upload a categories file.', '', '', '', {'display': 'none'}, None, {'display': 'none'}, False
        
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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return 'Error in BN step execution: ' + stderr.decode(), '', '', '', {'display': 'none'}, None, {'display': 'none'}, False
        
        clear_assets_folder()
        try:
            shutil.copy(os.path.join(output_folder, 'Surr_plot.png'), os.path.join('assets', 'Surr_plot.png'))
        except:
            print()
        
        #df_results = pd.read_csv(output_folder+"CCM_ECCM_curated.csv")       
        #visualize_network(df_results, output_folder+"network_plot.png")
        shutil.copy(os.path.join(output_folder, 'network_plot.png'), os.path.join('assets', 'network_plot.png'))        
        
        return 'BN step completed successfully.', '', '', '', {'display': 'none'}, [
            #instructions_text,
            present_all_density_ccm(output_folder),
            present_all_eccm(output_folder),
            html.Img(src='assets/network_plot.png'),
            html.Img(src='assets/Surr_plot.png'),
            present_all_bn_results(output_folder)], {'display': 'block'}, True

    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=False, port=8050, host='127.0.0.1')
