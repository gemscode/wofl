"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
let isAuthenticated = false;
function activate(context) {
    checkAuthenticationState(context);
    const mainProvider = new MainProvider(context);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider('rw-main', mainProvider));
    context.subscriptions.push(vscode.commands.registerCommand('rw-v1.openChat', () => vscode.commands.executeCommand('workbench.view.extension.rw-sidebar')), vscode.commands.registerCommand('rw-v1.optimize', () => handleCodeAction(context, 'optimize')), vscode.commands.registerCommand('rw-v1.explain', () => handleCodeAction(context, 'explain')), vscode.commands.registerCommand('rw-v1.debug', () => handleCodeAction(context, 'debug')));
}
async function checkAuthenticationState(context) {
    const token = await context.secrets.get('rw-jwt');
    isAuthenticated = !!token;
    vscode.commands.executeCommand('setContext', 'rw.authenticated', isAuthenticated);
}
async function handleCodeAction(context, action) {
    const editor = vscode.window.activeTextEditor;
    if (!editor)
        return;
    const selection = editor.selection;
    const text = editor.document.getText(selection);
    if (!text) {
        vscode.window.showWarningMessage('Please select code first');
        return;
    }
    const token = await context.secrets.get('rw-jwt');
    if (!token) {
        vscode.window.showErrorMessage('Authentication required');
        return;
    }
    const prompt = `${action} this code: ${text}`;
    vscode.commands.executeCommand('rw-v1.sendPrompt', prompt);
}
class MainProvider {
    context;
    webviewView;
    constructor(context) {
        this.context = context;
    }
    resolveWebviewView(webviewView) {
        this.webviewView = webviewView;
        webviewView.webview.options = { enableScripts: true };
        this.updateView();
        webviewView.webview.onDidReceiveMessage(this.handleMessage.bind(this));
        this.context.subscriptions.push(vscode.commands.registerCommand('rw-v1.sendPrompt', (prompt) => {
            webviewView.webview.postMessage({ type: 'setPrompt', prompt });
        }));
    }
    async updateView() {
        if (!this.webviewView)
            return;
        const token = await this.context.secrets.get('rw-jwt');
        if (token) {
            this.webviewView.webview.html = this.getChatHTML();
        }
        else {
            this.webviewView.webview.html = this.getLoginHTML();
        }
    }
    async handleMessage(message) {
        switch (message.command) {
            case 'login':
            case 'register':
                await this.handleAuth(message.command, message.email, message.password);
                break;
            case 'send-prompt':
                await this.handlePrompt(message.prompt);
                break;
            case 'logout':
                await this.handleLogout();
                break;
            case 'copy-code':
                await this.handleCopyCode(message.code);
                break;
        }
    }
    async handleAuth(action, email, password) {
        try {
            const endpoint = action === 'login' ? '/login' : '/register';
            const response = await fetch(`http://localhost:5001${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await response.json();
            if (response.ok && data.token) {
                await this.context.secrets.store('rw-jwt', data.token);
                isAuthenticated = true;
                vscode.commands.executeCommand('setContext', 'rw.authenticated', true);
                this.updateView();
                vscode.window.showInformationMessage('Authentication successful!');
            }
            else {
                this.webviewView?.webview.postMessage({
                    type: 'auth-error',
                    message: data.error || 'Authentication failed'
                });
            }
        }
        catch (error) {
            this.webviewView?.webview.postMessage({
                type: 'auth-error',
                message: 'Network error - check backend connection'
            });
        }
    }
    async handlePrompt(prompt) {
        const token = await this.context.secrets.get('rw-jwt');
        if (!token) {
            await this.context.secrets.delete('rw-jwt');
            isAuthenticated = false;
            vscode.commands.executeCommand('setContext', 'rw.authenticated', false);
            this.updateView();
            return;
        }
        this.webviewView?.webview.postMessage({ type: 'prompt-start' });
        try {
            const response = await fetch('http://localhost:5001/api/prompt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ prompt })
            });
            const data = await response.json();
            if (response.ok && data.response) {
                this.webviewView?.webview.postMessage({
                    type: 'prompt-response',
                    code: data.response
                });
            }
            else if (response.status === 401) {
                await this.handleLogout();
            }
            else {
                this.webviewView?.webview.postMessage({
                    type: 'prompt-error',
                    error: data.error || 'API error'
                });
            }
        }
        catch (error) {
            this.webviewView?.webview.postMessage({
                type: 'prompt-error',
                error: 'Network error'
            });
        }
    }
    async handleLogout() {
        await this.context.secrets.delete('rw-jwt');
        isAuthenticated = false;
        vscode.commands.executeCommand('setContext', 'rw.authenticated', false);
        this.updateView();
        vscode.window.showInformationMessage('Session expired. Please login again.');
    }
    async handleCopyCode(code) {
        await vscode.env.clipboard.writeText(code);
        vscode.window.showInformationMessage('Copied code to clipboard!');
    }
    getLoginHTML() {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    padding: 20px;
                    background: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    font-family: var(--vscode-font-family);
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                .login-container {
                    max-width: 300px;
                    margin: 0 auto;
                }
                h2 {
                    text-align: center;
                    margin-bottom: 30px;
                    color: var(--vscode-foreground);
                }
                .form-group {
                    margin-bottom: 15px;
                }
                input {
                    width: 100%;
                    padding: 10px;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    box-sizing: border-box;
                }
                button {
                    width: 100%;
                    padding: 12px;
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }
                button:hover {
                    background: var(--vscode-button-hoverBackground);
                }
                .toggle-mode {
                    margin-top: 15px;
                    text-align: center;
                    color: var(--vscode-textLink-foreground);
                    cursor: pointer;
                    text-decoration: underline;
                }
                .error {
                    color: var(--vscode-errorForeground);
                    margin-top: 10px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h2 id="title">Welcome to R&W</h2>
                <div class="form-group">
                    <input type="email" id="email" placeholder="Email" required>
                </div>
                <div class="form-group">
                    <input type="password" id="password" placeholder="Password" required>
                </div>
                <button onclick="handleAuth()">Login</button>
                <div class="error" id="error"></div>
                <div class="toggle-mode" onclick="toggleMode()">
                    Need an account? Register here
                </div>
            </div>
            <script>
                const vscode = acquireVsCodeApi();
                let isLogin = true;

                function toggleMode() {
                    isLogin = !isLogin;
                    document.getElementById('title').textContent = isLogin ? 'Welcome to R&W' : 'Create R&W Account';
                    document.querySelector('button').textContent = isLogin ? 'Login' : 'Register';
                    document.querySelector('.toggle-mode').textContent = isLogin 
                        ? 'Need an account? Register here' 
                        : 'Already have an account? Login';
                    document.getElementById('error').textContent = '';
                }

                function handleAuth() {
                    const email = document.getElementById('email').value;
                    const password = document.getElementById('password').value;
                    
                    if (!email || !password) {
                        document.getElementById('error').textContent = 'Please fill all fields';
                        return;
                    }

                    vscode.postMessage({
                        command: isLogin ? 'login' : 'register',
                        email: email,
                        password: password
                    });
                }

                window.addEventListener('message', event => {
                    const message = event.data;
                    if (message.type === 'auth-error') {
                        document.getElementById('error').textContent = message.message;
                    }
                });

                document.getElementById('email').focus();
            </script>
        </body>
        </html>`;
    }
    getChatHTML() {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    font-family: var(--vscode-font-family);
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                .header {
                    padding: 10px 15px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: var(--vscode-sideBar-background);
                }
                #monaco-container {
                    flex: 1;
                    min-height: 200px;
                }
                .button-bar {
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                    background: var(--vscode-editor-background);
                    padding: 8px 15px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
                .copy-btn {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 12px;
                    cursor: pointer;
                    transition: background 0.2s;
                    font-weight: 500;
                }
                .copy-btn:hover {
                    background: var(--vscode-button-hoverBackground);
                }
                .input-container {
                    padding: 15px;
                    background: var(--vscode-sideBar-background);
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .prompt-input {
                    width: 100%;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    padding: 10px;
                    font-size: 14px;
                    font-family: inherit;
                    box-sizing: border-box;
                    outline: none;
                    transition: border 0.2s;
                }
                .prompt-input:focus {
                    border: 1.5px solid var(--vscode-focusBorder, #007acc);
                }
                .input-row {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                .send-btn {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                    cursor: pointer;
                    white-space: nowrap;
                    font-size: 14px;
                }
                .send-btn:hover {
                    background: var(--vscode-button-hoverBackground);
                }
                .send-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .logout-btn {
                    background: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    border: none;
                    border-radius: 3px;
                    padding: 5px 10px;
                    cursor: pointer;
                    font-size: 12px;
                }
                .quick-actions {
                    display: flex;
                    gap: 5px;
                    flex-wrap: wrap;
                }
                .quick-action {
                    background: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    border: none;
                    border-radius: 3px;
                    padding: 5px 10px;
                    cursor: pointer;
                    font-size: 11px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <span>🤖 AI Assistant</span>
                <button class="logout-btn" onclick="logout()">Logout</button>
            </div>
            
            <div id="monaco-container"></div>
            
            <div class="button-bar">
                <button class="copy-btn" onclick="copyCode()">📋 Copy</button>
            </div>
            
            <div class="input-container">
                <div class="quick-actions">
                    <button class="quick-action" onclick="setPrompt('Optimize this code')">🚀 Optimize</button>
                    <button class="quick-action" onclick="setPrompt('Explain this code')">💡 Explain</button>
                    <button class="quick-action" onclick="setPrompt('Debug this code')">🐛 Debug</button>
                    <button class="quick-action" onclick="setPrompt('Add comments')">📝 Comment</button>
                </div>
                
                <div class="input-row">
                    <input
                        id="promptInput"
                        class="prompt-input"
                        type="text"
                        placeholder="Ask AI to optimize, explain, or debug your code..."
                        onkeydown="handleKeyDown(event)"
                        autocomplete="off"
                    />
                    <button id="sendBtn" class="send-btn" onclick="sendPrompt()">Submit</button>
                </div>
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.36.1/min/vs/loader.min.js"></script>
            <script>
                const vscode = acquireVsCodeApi();
                let isProcessing = false;
                let monacoEditor;

                require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.36.1/min/vs' }});
                require(['vs/editor/editor.main'], function() {
                    monacoEditor = monaco.editor.create(document.getElementById('monaco-container'), {
                        value: '# AI-generated code will appear here\\n',
                        language: 'python',
                        theme: 'vs-dark',
                        automaticLayout: true,
                        minimap: { enabled: false },
                        scrollBeyondLastLine: false,
                        fontSize: 14,
                        lineNumbers: 'on',
                        roundedSelection: false,
                        scrollbar: {
                            vertical: 'visible',
                            horizontal: 'visible'
                        }
                    });
                });

                function copyCode() {
                    if (monacoEditor) {
                        vscode.postMessage({ command: 'copy-code', code: monacoEditor.getValue() });
                    }
                }

                function handleKeyDown(event) {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        sendPrompt();
                    }
                }

                function setPrompt(text) {
                    document.getElementById('promptInput').value = text;
                    document.getElementById('promptInput').focus();
                }

                function sendPrompt() {
                    const input = document.getElementById('promptInput');
                    const prompt = input.value.trim();
                    
                    if (!prompt || isProcessing) return;

                    isProcessing = true;
                    document.getElementById('sendBtn').disabled = true;
                    document.getElementById('sendBtn').textContent = 'Sending...';

                    vscode.postMessage({
                        command: 'send-prompt',
                        prompt: prompt
                    });

                    input.value = '';
                }

                function logout() {
                    vscode.postMessage({ command: 'logout' });
                }

                window.addEventListener('message', event => {
                    const message = event.data;
                    switch (message.type) {
                        case 'setPrompt':
                            setPrompt(message.prompt);
                            break;
                        case 'prompt-start':
                            if (monacoEditor) monacoEditor.setValue('# Generating...');
                            break;
                        case 'prompt-response':
                            if (monacoEditor) monacoEditor.setValue(message.code);
                            finishProcessing();
                            break;
                        case 'prompt-error':
                            if (monacoEditor) monacoEditor.setValue('❌ Error: ' + message.error);
                            finishProcessing();
                            break;
                    }
                });

                function finishProcessing() {
                    isProcessing = false;
                    document.getElementById('sendBtn').disabled = false;
                    document.getElementById('sendBtn').textContent = 'Submit';
                }

                document.getElementById('promptInput').focus();
            </script>
        </body>
        </html>`;
    }
}
function deactivate() { }
//# sourceMappingURL=extension.js.map