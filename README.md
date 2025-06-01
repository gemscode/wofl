```markdown
# R&W AI Companion Framework

**Enterprise-Grade AI-Driven Application Development Platform**

---

## Overview

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-brightgreen)](https://github.com/gemscode/wofl.git)

A comprehensive framework for building AI-driven enterprise applications through modular agents and integrated tooling:

- **Core Agents**: Kafka, Redis, Kubernetes, Security, Storage (Cassandra/Elasticsearch), Docker
- **AI Middleware**: LangChain + DSPy integration with reinforcement learning workflows
- **VS Code Ecosystem**: Extension (rw_vscode) + backend services (vscbackend) for seamless development
- **CLI Power**: Create/test/deploy full-stack applications with single commands

---

## System Architecture

```
wofl/
├── rw_agent/          # Core framework and CLI
├── rw_vscode/         # VS Code extension
├── vscbackend/        # Extension backend services
├── requirements.txt   # Python dependencies
└── init_rw_agent.sh   # Bootstrap script
```

---

## Getting Started

### Requirements
- Python 3.9+
- Node.js 16+ (for VS Code extension)
- Docker (recommended)

### Installation

1. **Clone repository and setup environment**

    ```
    git clone https://github.com/gemscode/wofl.git
    cd wofl

    # Create and activate virtual environment (outside project dir recommended)
    python3 -m venv ~/venvs/rwagent
    source ~/venvs/rwagent/bin/activate

    # Install core requirements
    pip install -r requirements.txt

    # Install framework in development mode
    cd rw_agent
    pip install -e .
    ```

2. **Initialize Sample Project**

    ```
    rwagent init-project --name my_enterprise_app
    cd my_enterprise_app
    ```

---

## Configuration

### Essential .env Setup

After project initialization, create `.env` in your project root:

```
# Required for core services
CASSANDRA_HOST=localhost
ELASTICSEARCH_URL=http://localhost:9200
REDIS_URL=redis://localhost:6379

# Optional (handled by VS Code extension if using integrated services)
# CLAUDE_API_KEY=your_key_here
# LLAMA_API_KEY=your_key_here
```

**Never commit .env to version control!**

---

## VS Code Integration

1. **Install Extension**

    ```
    cd wofl/rw_vscode
    npm install
    npm run package
    code --install-extension rw-vscode-0.1.0.vsix
    ```

2. **Key Features**
    - AI-assisted code generation (no external API keys needed for core features)
    - Real-time infrastructure monitoring
    - One-click deployment pipelines
    - Integrated agent debugging

---

## Core Workflows

### CLI Operations

```
# From project root
rwagent check-integrity    # Validate project structure
rwagent fix-integrity     # Repair configuration issues
rwagent deploy-service    # Deploy to configured environment
```

### Extension Usage
1. Open project in VS Code
2. Command Palette (`Ctrl+Shift+P`):
   - `RW: New AI Feature`
   - `RW: Validate Deployment`
   - `RW: Monitor Agents`

---

## Development Notes

### Virtual Environments
- Keep venv **outside** project directory
- Recreate using:
    ```
    python3 -m venv /path/to/venv
    source /path/to/venv/bin/activate
    pip install -r wofl/requirements.txt
    ```

### Contributing
1. Fork repository
2. Branch structure:
   - `rw_agent/`: Core framework changes
   - `rw_vscode/`: VS Code extension updates
   - `vscbackend/`: Extension backend services
3. Submit PR with clear component labeling

---

## License
Apache 2.0 - See [LICENSE](LICENSE)
```

