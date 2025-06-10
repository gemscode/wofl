<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>R&amp;W AI Companion Framework - README</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }
    h1, h2, h3 { color: #2c3e50; }
    code, pre { background: #f4f4f4; border-radius: 4px; padding: 2px 6px; }
    pre { padding: 1em; overflow-x: auto; }
    blockquote { border-left: 4px solid #ccc; margin: 1em 0; padding-left: 1em; color: #555; }
    ul, ol { margin-left: 2em; }
    .badge { vertical-align: middle; }
    hr { margin: 2em 0; }
  </style>
</head>
<body>

<h1>R&amp;W AI Companion Framework</h1>
<p><strong>Enterprise-Grade AI-Driven Application Development Platform</strong></p>

<hr>

<h2>Overview</h2>

<p>
  <a href="https://github.com/gemscode/wofl.git">
    <img src="https://img.shields.io/badge/GitHub-Repository-brightgreen" alt="GitHub Repository" class="badge">
  </a>
</p>

<p>
  A comprehensive framework for building AI-driven enterprise applications through modular agents and integrated tooling:
</p>
<ul>
  <li><strong>Core Agents:</strong> Kafka, Redis, Kubernetes, Security, Storage (Cassandra/Elasticsearch), Docker</li>
  <li><strong>AI Middleware:</strong> LangChain + DSPy integration with reinforcement learning workflows</li>
  <li><strong>VS Code Ecosystem:</strong> Extension (rw_vscode) + backend services (vscbackend) for seamless development</li>
  <li><strong>CLI Power:</strong> Create/test/deploy full-stack applications with single commands</li>
</ul>

<hr>

<h2>System Architecture</h2>

<pre>
wofl/
├── deployments/         # Service deployment configurations (Cassandra, Elasticsearch, etc.)
├── framework/           # Framework initialization and virtual environment and CLI
├── rw_agent/            # Core agent framework
├── rw_vscode/           # VS Code extension
├── vscbackend/          # Extension backend services
├── core_requirements.in # Python dependencies
└── install.sh           # Main installation script
</pre>

<hr>

<h2>Getting Started</h2>

<h3>Requirements</h3>
<ul>
  <li>Python 3.9+</li>
  <li>Node.js 16+ (for VS Code extension)</li>
  <li>Docker (recommended)</li>
</ul>

<h3>Installation</h3>

<ol>
  <li>
    <strong>Clone repository and setup environment</strong>
    <pre>
git clone https://github.com/gemscode/wofl.git
cd wofl

# Create and activate virtual environment (outside project dir recommended)
python3 -m venv ~/venvs/rwagent
source ~/venvs/rwagent/bin/activate

# Install core requirements for framework and cli rwagent
./install.sh (you might need to chmod +x ./install.sh)
    </pre>
  </li>
  <li>
    <strong>Initialize Sample Project</strong>
    <pre>
rwagent init-project --name my_enterprise_app
cd my_enterprise_app
    </pre>
  </li>
</ol>

<hr>

<h2>Configuration</h2>

<h3>Essential .env Setup</h3>
<p>After project initialization, create <code>.env</code> in your project root:</p>
<pre>
# Required for core services
CASSANDRA_HOST=localhost
ELASTICSEARCH_URL=http://localhost:9200
REDIS_URL=redis://localhost:6379

# Optional (handled by VS Code extension if using integrated services)
# CLAUDE_API_KEY=your_key_here
# LLAMA_API_KEY=your_key_here
</pre>
<p><strong>Never commit .env to version control!</strong></p>

<hr>

<h2>VS Code Integration</h2>

<ol>
  <li>
    <strong>Install Extension</strong>
    <pre>
cd wofl/rw_vscode
npm install
npm run package
code --install-extension rw-vscode-0.1.0.vsix
    </pre>
  </li>
  <li>
    <strong>Key Features</strong>
    <ul>
      <li>AI-assisted code generation (no external API keys needed for core features)</li>
      <li>Real-time infrastructure monitoring</li>
      <li>One-click deployment pipelines</li>
      <li>Integrated agent debugging</li>
    </ul>
  </li>
</ol>

<hr>

<h2>Core Workflows</h2>

<h3>CLI Operations</h3>
<pre>
# From project root
rwagent check-integrity    # Validate project structure
rwagent fix-integrity     # Repair configuration issues
rwagent deploy-service    # Deploy to configured environment
</pre>

<h3>Extension Usage</h3>
<ol>
  <li>Open project in VS Code</li>
  <li>Command Palette (<code>Ctrl+Shift+P</code>):
    <ul>
      <li>RW: New AI Feature</li>
      <li>RW: Validate Deployment</li>
      <li>RW: Monitor Agents</li>
    </ul>
  </li>
</ol>

<hr>

<h2>Development Notes</h2>

<h3>Virtual Environments</h3>
<ul>
  <li>Keep venv <strong>outside</strong> project directory</li>
  <li>Recreate using:
    <pre>
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install -r wofl/requirements.txt
    </pre>
  </li>
</ul>

<h3>Contributing</h3>
<ol>
  <li>Fork repository</li>
  <li>Branch structure:
    <ul>
      <li><code>rw_agent/</code>: Core framework changes</li>
      <li><code>rw_vscode/</code>: VS Code extension updates</li>
      <li><code>vscbackend/</code>: Extension backend services</li>
    </ul>
  </li>
  <li>Submit PR with clear component labeling</li>
</ol>

<hr>

<h2>License</h2>
<p>
  Apache 2.0 - See <a href="LICENSE">LICENSE</a>
</p>

</body>
</html>

