import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

interface AuthResponse {
    token?: string;
    error?: string;
}

interface PromptResponse {
    response?: string;
    thread_id?: string;
    error?: string;
    new_thread?: boolean;
}

interface ProjectFile {
    name: string;
    path: string;
    type: 'file' | 'folder';
}

let isAuthenticated = false;
let currentThreadId: string | null = null;
let newThreadFlag = true;

export function activate(context: vscode.ExtensionContext) {
    checkAuthenticationState(context);

    const mainProvider = new MainProvider(context);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('rw-main', mainProvider)
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('rw-v1.openChat', () =>
            vscode.commands.executeCommand('workbench.view.extension.rw-sidebar')),
        vscode.commands.registerCommand('rw-v1.optimize', () => handleCodeAction(context, 'optimize')),
        vscode.commands.registerCommand('rw-v1.explain', () => handleCodeAction(context, 'explain')),
        vscode.commands.registerCommand('rw-v1.debug', () => handleCodeAction(context, 'debug')),
        vscode.commands.registerCommand('rw-v1.newThread', () => {
            currentThreadId = null;
            newThreadFlag = true;
            mainProvider.updateView();
            vscode.window.showInformationMessage('New thread started');
        })
    );
}

async function checkAuthenticationState(context: vscode.ExtensionContext) {
    const token = await context.secrets.get('rw-jwt');
    isAuthenticated = !!token;
    vscode.commands.executeCommand('setContext', 'rw.authenticated', isAuthenticated);
}

async function handleCodeAction(context: vscode.ExtensionContext, action: string) {
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

class MainProvider implements vscode.WebviewViewProvider {
    private webviewView?: vscode.WebviewView;

    constructor(private context: vscode.ExtensionContext) {}

    resolveWebviewView(webviewView: vscode.WebviewView) {
        this.webviewView = webviewView;
        webviewView.webview.options = { 
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(this.context.extensionPath, 'src', 'webviews'))]
        };

        this.updateView();
        webviewView.webview.onDidReceiveMessage(this.handleMessage.bind(this));
        
        this.context.subscriptions.push(
            vscode.commands.registerCommand('rw-v1.sendPrompt', (prompt: string) => {
                webviewView.webview.postMessage({ type: 'setPrompt', prompt });
            })
        );
    }

    public async updateView() {
        if (!this.webviewView) {
            return;
        }

        const token = await this.context.secrets.get('rw-jwt');
        if (token) {
            this.webviewView.webview.html = this.getWebviewContent('chat.html');
            // Send initial file list
            const files = await this.getProjectFiles();
            this.webviewView.webview.postMessage({ type: 'file-list', files });
        } else {
            this.webviewView.webview.html = this.getWebviewContent('login.html');
        }
    }

    private async getProjectFiles(): Promise<ProjectFile[]> {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return [];
        }
        
        const files: ProjectFile[] = [];
        
        for (const folder of workspaceFolders) {
            // Look for rw_agent structure
            const agentPath = path.join(folder.uri.fsPath, 'src', 'agents');
            if (fs.existsSync(agentPath)) {
                this.scanDirectory(agentPath, files, 'agents');
            }
        }
        
        return files;
    }

    private scanDirectory(dirPath: string, files: ProjectFile[], prefix: string) {
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
                } else if (entry.name.endsWith('.py')) {
                    files.push({
                        name: relativeName,
                        path: fullPath,
                        type: 'file'
                    });
                }
            }
        } catch (error) {
            console.error('Error scanning directory:', error);
        }
    }

    private async handleMessage(message: any) {
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

    private getWebviewContent(htmlFile: string): string {
        const htmlPath = path.join(this.context.extensionPath, 'src', 'webviews', htmlFile);
        let html = fs.readFileSync(htmlPath, 'utf8');
        
        // Replace resource URIs
        const webview = this.webviewView!.webview;
        const stylesUri = webview.asWebviewUri(vscode.Uri.file(
            path.join(this.context.extensionPath, 'src', 'webviews', 'styles.css')
        ));
        
        html = html.replace('{{stylesUri}}', stylesUri.toString());
        return html;
    }

    // ... (rest of the existing methods: handleAuth, handlePrompt, handleLogout, handleCopyCode)
    private async handleAuth(action: string, email: string, password: string) {
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

            const data = await response.json() as AuthResponse;

            if (response.ok && data.token) {
                await this.context.secrets.store('rw-jwt', data.token);
                isAuthenticated = true;
                vscode.commands.executeCommand('setContext', 'rw.authenticated', true);
                this.updateView();
                vscode.window.showInformationMessage('Authentication successful!');
            } else {
                this.webviewView?.webview.postMessage({
                    type: 'auth-error',
                    message: data.error || 'Authentication failed'
                });
            }
        } catch (error) {
            this.webviewView?.webview.postMessage({
                type: 'auth-error',
                message: 'Network error - check backend connection'
            });
        }
    }

    private async handlePrompt(prompt: string) {
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

            const data = await response.json() as PromptResponse;

            if (response.ok && data.response) {
                currentThreadId = data.thread_id || null;
                newThreadFlag = false;
                this.webviewView?.webview.postMessage({
                    type: 'prompt-response',
                    code: data.response,
                    threadId: data.thread_id,
                    isNewThread: data.new_thread
                });
            } else if (response.status === 401) {
                await this.handleLogout();
            } else {
                this.webviewView?.webview.postMessage({
                    type: 'prompt-error',
                    error: data.error || 'API error'
                });
            }
        } catch (error) {
            this.webviewView?.webview.postMessage({
                type: 'prompt-error',
                error: 'Network error'
            });
        }
    }

    private async handleLogout() {
        await this.context.secrets.delete('rw-jwt');
        isAuthenticated = false;
        currentThreadId = null;
        newThreadFlag = true;
        vscode.commands.executeCommand('setContext', 'rw.authenticated', false);
        this.updateView();
        vscode.window.showInformationMessage('Session expired. Please login again.');
    }

    private async handleCopyCode(code: string) {
        await vscode.env.clipboard.writeText(code);
        vscode.window.showInformationMessage('Copied code to clipboard!');
    }
}

export function deactivate() {}

