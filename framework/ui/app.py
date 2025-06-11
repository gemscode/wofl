import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import dash_monaco_editor
import requests
import uuid
from flask import Flask, session
import os

# --------- FILE SYSTEM CONFIGURATION ---------
# Calculate the root directory relative to this file
current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..', '..'))
AGENT_DIR = os.path.join(root_dir, 'framework', 'rw_agent', 'src', 'agents')

server = Flask(__name__)
server.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

def initialize_session():
    defaults = {
        'authenticated': False,
        'auth_token': None,
        'current_thread': None,
        'new_thread_flag': True,
        'code_output': "# AI-generated code will appear here\n",
        'current_file': None
    }
    for k, v in defaults.items():
        session.setdefault(k, v)

def handle_authentication(action, email, password):
    endpoint = "/rw/login" if action == "login" else "/rw/register"
    try:
        resp = requests.post(
            f"https://www.wolfx0.com{endpoint}",
            headers={"Content-Type": "application/json", "Origin": "vscode-webview://"},
            json={"email": email, "password": password}
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("token"):
                session['auth_token'] = data["token"]
                session['authenticated'] = True
                return True
        return False
    except Exception as e:
        print(f"Auth error: {e}")
        return False

def get_agent_files():
    """Get files from local filesystem instead of API"""
    files = []
    
    # Add "New File" option at the top
    files.append("new_file")
    
    try:
        if os.path.exists(AGENT_DIR):
            for root, dirs, file_list in os.walk(AGENT_DIR):
                for file in file_list:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, AGENT_DIR)
                        files.append(rel_path)
        print(f"Found {len(files)} files in {AGENT_DIR}")
    except Exception as e:
        print(f"Error reading agent directory: {e}")
    
    return files

def read_file_content(file_path):
    """Read content from a local file"""
    try:
        full_path = os.path.join(AGENT_DIR, file_path)
        if os.path.exists(full_path) and file_path.endswith('.py'):
            with open(full_path, 'r') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return ""

def save_file_content(file_path, content):
    """Save content to a local file"""
    try:
        full_path = os.path.join(AGENT_DIR, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        return False

def handle_prompt_submission(prompt):
    try:
        resp = requests.post(
            "https://wolfx0.com/rw/prompt",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {session.get('auth_token')}",
                "Origin": "vscode-webview://"
            },
            json={
                "prompt": prompt,
                "thread_id": session.get('current_thread'),
                "new_thread": session.get('new_thread_flag', True)
            }
        )
        if resp.status_code == 200:
            data = resp.json()
            session['current_thread'] = data.get("thread_id")
            session['new_thread_flag'] = False
            session['code_output'] = data.get("response", "")
            return True, data.get("response", "")
        elif resp.status_code == 401:
            session.clear()
            return False, "Session expired - please login again"
        else:
            return False, f"API Error: {resp.json().get('error', 'Unknown error')}"
    except Exception as e:
        return False, f"Connection error: {e}"

def create_login_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("R&W AI Companion", className="text-center mb-4", style={'fontWeight': 700, 'color': '#2c3e50'}),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Login", tab_id="login"),
                            dbc.Tab(label="Register", tab_id="register"),
                        ], id="auth-tabs", active_tab="login"),
                        html.Div(id="auth-content", className="mt-4")
                    ])
                ], style={'border': '1px solid #dee2e6', 'borderRadius': '8px'})
            ], width=6)
        ], justify="center", className="mt-5")
    ], fluid=True)

def create_login_form():
    return dbc.Form([
        dbc.Input(type="email", id="login-email", placeholder="Email", className="mb-3"),
        dbc.Input(type="password", id="login-password", placeholder="Password", className="mb-3"),
        dbc.Button("Login", id="login-btn", color="primary", className="w-100 mb-2", style={'height': '42px', 'fontWeight': '500'}),
        html.Div(id="login-feedback")
    ])

def create_register_form():
    return dbc.Form([
        dbc.Input(type="email", id="register-email", placeholder="Email", className="mb-3"),
        dbc.Input(type="password", id="register-password", placeholder="Password", className="mb-3"),
        dbc.Input(type="password", id="register-confirm", placeholder="Confirm Password", className="mb-3"),
        dbc.Button("Register", id="register-btn", color="success", className="w-100 mb-2", style={'height': '42px', 'fontWeight': '500'}),
        html.Div(id="register-feedback")
    ])

def create_main_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("R&W AI Companion", className="mb-4", style={'fontWeight': 700, 'color': '#2c3e50'}))
        ], className="mt-3"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Dropdown(
                        id="file-dropdown",
                        options=[],
                        placeholder="Select Agent File",
                        style={'width': '220px', 'marginRight': '12px', 'fontSize': '14px'}
                    ),
                    dbc.Button(
                        "🧵 New Thread",
                        id="new-thread-btn",
                        color="primary",
                        style={'width': '140px', 'height': '38px', 'marginRight': '12px'}
                    ),
                    html.Div(
                        dbc.Button(
                            "Sign Out",
                            id="logout-btn",
                            color="danger",
                            style={'width': '120px', 'height': '38px'}
                        ),
                        style={'marginLeft': 'auto', 'display': 'flex', 'flex': 1, 'justifyContent': 'flex-end'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '24px'}),
                dash_monaco_editor.DashMonacoEditor(
                    id="code-editor",
                    value=session.get('code_output', "# AI-generated code will appear here\n"),
                    language="python",
                    theme="vs-dark",
                    height="400px"
                ),
                html.Div([
                    dbc.Button("🔨 Build", id="build-btn", color="primary", style={'width': '120px', 'marginRight': '8px'}),
                    dbc.Button("▶️ Run", id="run-btn", color="primary", style={'width': '120px', 'marginRight': '8px'}),
                    dbc.Button("🚀 Deploy", id="deploy-btn", color="primary", style={'width': '120px'}),
                    dbc.Button("💾 Save", id="save-btn", color="success", style={'width': '120px', 'marginLeft': '20px'}, disabled=True)
                ], style={'display': 'flex', 'margin': '20px 0'}),
                html.Div([
                    dcc.Input(
                        id="prompt-input",
                        placeholder="Type your coding prompt here...",
                        style={'flex': 1, 'marginRight': '12px', 'height': '38px', 'borderRadius': '4px', 'border': '1px solid #ced4da', 'padding': '0 12px'}
                    ),
                    dbc.Button(
                        "Submit",
                        id="submit-btn",
                        color="primary",
                        style={'width': '120px', 'height': '38px'},
                        disabled=True
                    )
                ], style={'display': 'flex', 'alignItems': 'center'})
            ])
        ], style={'marginBottom': '40px'})
    ], fluid=True, style={
        'maxWidth': '1400px',
        'margin': '40px auto 0 auto',
        'background': '#fff',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.04)',
        'padding': '32px 32px 24px 32px'
    })

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='current-file-store'),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    initialize_session()
    if session.get('authenticated'):
        return create_main_layout()
    else:
        return create_login_layout()

@app.callback(
    Output('auth-content', 'children'),
    Input('auth-tabs', 'active_tab')
)
def render_auth_content(active_tab):
    if active_tab == "login":
        return create_login_form()
    else:
        return create_register_form()

@app.callback(
    [Output('login-feedback', 'children'),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('login-btn', 'n_clicks'),
    State('login-email', 'value'),
    State('login-password', 'value'),
    prevent_initial_call=True
)
def handle_login(n_clicks, email, password):
    if n_clicks and email and password:
        if handle_authentication("login", email, password):
            return dbc.Alert("Login successful!", color="success", dismissible=True), "/"
        else:
            return dbc.Alert("Invalid credentials", color="danger", dismissible=True), dash.no_update
    return "", dash.no_update

@app.callback(
    [Output('register-feedback', 'children'),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('register-btn', 'n_clicks'),
    State('register-email', 'value'),
    State('register-password', 'value'),
    State('register-confirm', 'value'),
    prevent_initial_call=True
)
def handle_register(n_clicks, email, password, confirm):
    if n_clicks and email and password and confirm:
        if password != confirm:
            return dbc.Alert("Passwords do not match", color="danger", dismissible=True), dash.no_update
        if handle_authentication("register", email, password):
            return dbc.Alert("Registration successful! Please login", color="success", dismissible=True), "/"
        else:
            return dbc.Alert("Registration failed", color="danger", dismissible=True), dash.no_update
    return "", dash.no_update

@app.callback(
    Output('file-dropdown', 'options'),
    Input('page-content', 'children')
)
def update_file_dropdown(_):
    if session.get('authenticated'):
        files = get_agent_files()
        options = []
        for file in files:
            if file == "new_file":
                options.append({'label': '➕ New File', 'value': 'new_file'})
            else:
                # Show directory structure in label
                label = file.replace('/', ' / ') if '/' in file else file
                options.append({'label': label, 'value': file})
        return options
    return []

@app.callback(
    Output('submit-btn', 'disabled'),
    Input('prompt-input', 'value')
)
def update_submit_button_state(prompt_value):
    return not (prompt_value and prompt_value.strip())

@app.callback(
    [Output('code-editor', 'value'),
     Output('prompt-input', 'value'),
     Output('current-file-store', 'data'),
     Output('save-btn', 'disabled')],
    [Input('submit-btn', 'n_clicks'),
     Input('new-thread-btn', 'n_clicks'),
     Input('build-btn', 'n_clicks'),
     Input('run-btn', 'n_clicks'),
     Input('deploy-btn', 'n_clicks'),
     Input('file-dropdown', 'value'),
     Input('save-btn', 'n_clicks')],
    [State('prompt-input', 'value'),
     State('code-editor', 'value'),
     State('current-file-store', 'data')],
    prevent_initial_call=True
)
def handle_actions(submit_clicks, new_thread_clicks, build_clicks, run_clicks, deploy_clicks, 
                   file_selected, save_clicks, prompt, current_code, current_file):
    triggered = ctx.triggered_id
    
    if triggered == 'file-dropdown':
        if file_selected == 'new_file':
            return "", "", "new_file", True
        elif file_selected:
            content = read_file_content(file_selected)
            return content, dash.no_update, file_selected, False
    
    elif triggered == 'save-btn' and current_file and current_file != 'new_file':
        success = save_file_content(current_file, current_code)
        if success:
            return dash.no_update, dash.no_update, dash.no_update, True
        
    elif triggered == 'submit-btn' and prompt:
        success, response = handle_prompt_submission(prompt)
        if success:
            return response, '', dash.no_update, current_file != 'new_file' and current_file is not None
        else:
            return current_code, prompt, dash.no_update, dash.no_update
            
    elif triggered == 'new-thread-btn':
        session.update({
            'current_thread': None,
            'new_thread_flag': True,
            'code_output': "# New thread started...\n"
        })
        return "# New thread started...\n", dash.no_update, dash.no_update, True
        
    elif triggered == 'build-btn':
        return current_code + "\n# Build process initiated...", dash.no_update, dash.no_update, dash.no_update
    elif triggered == 'run-btn':
        return current_code + "\n# Application running...", dash.no_update, dash.no_update, dash.no_update
    elif triggered == 'deploy-btn':
        return current_code + "\n# Deployment started...", dash.no_update, dash.no_update, dash.no_update
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('logout-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_logout(n_clicks):
    if n_clicks:
        session.clear()
        return "/"
    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

