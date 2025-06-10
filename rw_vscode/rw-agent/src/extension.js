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
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
let isAuthenticated = false;
let currentThreadId = null;
let newThreadFlag = true;
function activate(context) {
    checkAuthenticationState(context);
    const mainProvider = new MainProvider(context);
    context.subscriptions.push(vscode.window.registerWebviewViewProvider('rw-main', mainProvider));
    context.subscriptions.push(vscode.commands.registerCommand('rw-v1.openChat', () => vscode.commands.executeCommand('workbench.view.extension.rw-sidebar')), vscode.commands.registerCommand('rw-v1.optimize', () => handleCodeAction(context, 'optimize')), vscode.commands.registerCommand('rw-v1.explain', () => handleCodeAction(context, 'explain')), vscode.commands.registerCommand('rw-v1.debug', () => handleCodeAction(context, 'debug')), vscode.commands.registerCommand('rw-v1.newThread', () => {
        currentThreadId = null;
        newThreadFlag = true;
        mainProvider.updateView();
        vscode.window.showInformationMessage('New thread started');
    }));
}
async function checkAuthenticationState(context) {
    const token = await context.secrets.get('rw-jwt');
    isAuthenticated = !!token;
    vscode.commands.executeCommand('setContext', 'rw.authenticated', isAuthenticated);
}
async function handleCodeAction(context, action) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return;
    }
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
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(this.context.extensionPath, 'src', 'webviews'))]
        };
        this.updateView();
        webviewView.webview.onDidReceiveMessage(this.handleMessage.bind(this));
        this.context.subscriptions.push(vscode.commands.registerCommand('rw-v1.sendPrompt', (prompt) => {
            webviewView.webview.postMessage({ type: 'setPrompt', prompt });
        }));
    }
    async updateView() {
        if (!this.webviewView) {
            return;
        }
        const token = await this.context.secrets.get('rw-jwt');
        if (token) {
            this.webviewView.webview.html = this.getWebviewContent('chat.html');
            // Send initial file list
            const files = await this.getProjectFiles();
            this.webviewView.webview.postMessage({ type: 'file-list', files });
        }
        else {
            this.webviewView.webview.html = this.getWebviewContent('login.html');
        }
    }
    async getProjectFiles() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return [];
        }
        const files = [];
        for (const folder of workspaceFolders) {
            // Look for rw_agent structure
            const agentPath = path.join(folder.uri.fsPath, 'src', 'agents');
            if (fs.existsSync(agentPath)) {
                this.scanDirectory(agentPath, files, 'agents');
            }
        }
        return files;
    }
    scanDirectory(dirPath, files, prefix) {
        try {
            const entries = fs.readdirSync(dirPath, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(dirPath, entry.name);
                const relativeName = `${prefix}/${entry.name}`;
                if (entry.isDirectory()) {
                    files.push({
                        name: relativeName,
                        path: fullPath,
                        type: 'folder'
                    });
                    this.scanDirectory(fullPath, files, relativeName);
                }
                else if (entry.name.endsWith('.py')) {
                    files.push({
                        name: relativeName,
                        path: fullPath,
                        type: 'file'
                    });
                }
            }
        }
        catch (error) {
            console.error('Error scanning directory:', error);
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
            case 'new-thread':
                currentThreadId = null;
                newThreadFlag = true;
                this.webviewView?.webview.postMessage({ type: 'thread-reset' });
                break;
            case 'open-file':
                if (message.path) {
                    const uri = vscode.Uri.file(message.path);
                    await vscode.window.showTextDocument(uri);
                }
                break;
            case 'request-files':
                const files = await this.getProjectFiles();
                this.webviewView?.webview.postMessage({ type: 'file-list', files });
                break;
        }
    }
    getWebviewContent(htmlFile) {
        const htmlPath = path.join(this.context.extensionPath, 'src', 'webviews', htmlFile);
        let html = fs.readFileSync(htmlPath, 'utf8');
        // Replace resource URIs
        const webview = this.webviewView.webview;
        const stylesUri = webview.asWebviewUri(vscode.Uri.file(path.join(this.context.extensionPath, 'src', 'webviews', 'styles.css')));
        html = html.replace('{{stylesUri}}', stylesUri.toString());
        return html;
    }
    // ... (rest of the existing methods: handleAuth, handlePrompt, handleLogout, handleCopyCode)
    async handleAuth(action, email, password) {
        try {
            const endpoint = action === 'login' ? '/rw/login' : '/rw/register';
            const response = await fetch(`https://www.wolfx0.com${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Origin': 'vscode-webview://'
                },
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
            const response = await fetch('https://wolfx0.com/rw/prompt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                    'Origin': 'vscode-webview://'
                },
                body: JSON.stringify({
                    prompt,
                    thread_id: currentThreadId,
                    new_thread: newThreadFlag
                })
            });
            const data = await response.json();
            if (response.ok && data.response) {
                currentThreadId = data.thread_id || null;
                newThreadFlag = false;
                this.webviewView?.webview.postMessage({
                    type: 'prompt-response',
                    code: data.response,
                    threadId: data.thread_id,
                    isNewThread: data.new_thread
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
        currentThreadId = null;
        newThreadFlag = true;
        vscode.commands.executeCommand('setContext', 'rw.authenticated', false);
        this.updateView();
        vscode.window.showInformationMessage('Session expired. Please login again.');
    }
    async handleCopyCode(code) {
        await vscode.env.clipboard.writeText(code);
        vscode.window.showInformationMessage('Copied code to clipboard!');
    }
}
function deactivate() { }
//# sourceMappingURL=extension.js.map