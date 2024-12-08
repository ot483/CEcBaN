import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import os
import pickle
import bnlearn as bn
import base64
import io

model = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("CEcBaN's BN Model Inferencer"),
            dcc.Upload(
                id='upload-files',
                children=html.Div(['Drag and Drop or ', html.A('Select the bnlearn_model.pkl and dict_model_essentials.pickle Files')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=True
            ),
            dcc.Store(id='stored-data'),  
            html.Div(id='model-stats', style={'margin-top': '20px'}),
            html.Div(id='input-form', style={'margin-top': '20px'}),
            dbc.Button('Run Inference', id='run-inference', n_clicks=0, color='primary', style={'margin-top': '20px'}),
            html.Div(id='model-output', style={'margin-top': '20px'})
        ], width=12),
    ]),
])

def decode_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded, filename

@app.callback(
    [Output('model-stats', 'children'), Output('input-form', 'children'), Output('stored-data', 'data')],
    [Input('upload-files', 'contents')],
    [State('upload-files', 'filename')]
)
def load_files(contents, filenames):
    global model  

    if contents is None:
        return '', '', {}

    dict_model_essentials = None

    for content, filename in zip(contents, filenames):
        decoded, filename = decode_contents(content, filename)
        if 'bnlearn_model.pkl' in filename:
            model_file = io.BytesIO(decoded)
            model = pickle.load(model_file) 
        elif 'dict_model_essentials.pickle' in filename:
            dict_model_essentials = pickle.loads(decoded)

    if model is None or dict_model_essentials is None:
        return "One or more required files are missing", '', {}

    stored_data = {
        'dict_model_essentials': dict_model_essentials
    }

    # Display model stats
    stats = [
        html.H3("Model Statistics"),
        html.P(f"Nodes: {len(dict_model_essentials['nodes'])}"),
        html.P(f"Target: {dict_model_essentials['target']}"),
        html.P(f"Accuracy: {dict_model_essentials['accuracy']}"),
        html.P(f"ROC AUC: {dict_model_essentials['roc_auc']}"),
    ]

    # Create input fields arranged horizontally allowing only 0, 1, or 2
    input_fields = []
    for node in dict_model_essentials['nodes']:
        input_fields.append(
            dbc.Col([
                html.Label(f"{node}"),
                dcc.Input(id={'type': 'input', 'index': node}, type='number', min=0, max=2, step=1, value=1)
            ], width="auto", style={'margin': '5px'})
        )

    return stats, [dbc.Row(input_fields)], stored_data

@app.callback(
    Output('model-output', 'children'),
    Input('run-inference', 'n_clicks'),
    [State('stored-data', 'data'), State({'type': 'input', 'index': ALL}, 'value')]
)
def run_inference(n_clicks, stored_data, values):
    if n_clicks == 0 or not stored_data:
        return ''

    dict_model_essentials = stored_data.get('dict_model_essentials')

    evidence = {}
    for node, value in zip(dict_model_essentials['nodes'], values):
        try:
            evidence[node] = int(value)
        except Exception as e:
            print(e)
    
    print("Evidence:", evidence)

    # Perform inference using the global model
    global model
    print(evidence)
    print("XXXXXX")
    print(dict_model_essentials['target'])
    try:
        q1 = bn.inference.fit(model, variables=[dict_model_essentials['target']], evidence=evidence)
        prediction_low = q1.df.p[0]
        prediction_high = q1.df.p[1]
    except IndexError as e:
        print(f"IndexError during inference: {e}")
        return html.Div([
            html.H3("Model Output"),
            html.P(f"Error during inference: {e}"),
        ])

    print(f"Predictions - Low: {prediction_low}, High: {prediction_high}")

    return html.Div([
        html.H3("Model Output"),
        html.P(f"Probability to get low: {prediction_low:.4f}"),
        html.P(f"Probability to get high: {prediction_high:.4f}"),
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
