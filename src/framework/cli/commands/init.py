import click
from pathlib import Path
from uuid import uuid4
from jinja2 import Template
import json
import questionary
import os
import subprocess

def validate_project_name(ctx, param, value):
    if not value.isidentifier():
        raise click.BadParameter("Project name must be a valid Python identifier")
    return value

def create_cassandra_tables():
    """Create all required Cassandra tables if agent_storage is enabled"""
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider

    cassandra_host = os.environ.get("CASSANDRA_HOST", "localhost")
    cassandra_keyspace = os.environ.get("CASSANDRA_KEYSPACE", "rwagent")
    cassandra_user = os.environ.get("CASSANDRA_USER", "")
    cassandra_pass = os.environ.get("CASSANDRA_PASS", "")

    if cassandra_user and cassandra_pass:
        cluster = Cluster([cassandra_host], auth_provider=PlainTextAuthProvider(username=cassandra_user, password=cassandra_pass))
    else:
        cluster = Cluster([cassandra_host])

    session = cluster.connect()
    # Create keyspace if not exists
    session.execute(f"""
    CREATE KEYSPACE IF NOT EXISTS {cassandra_keyspace}
    WITH replication = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }}
    """)
    session.set_keyspace(cassandra_keyspace)

    # Create agent_metadata table
    session.execute("""
    CREATE TABLE IF NOT EXISTS agent_metadata (
        agent_id UUID,
        type TEXT,
        key TEXT,
        value TEXT,
        last_updated TIMESTAMP,
        PRIMARY KEY ((agent_id), type, key)
    ) WITH CLUSTERING ORDER BY (type ASC, key ASC)
    """)

    # Other necessary tables (optional)
    session.execute("""
    CREATE TABLE IF NOT EXISTS agent_registry (
        agent_id UUID PRIMARY KEY,
        name TEXT,
        description TEXT,
        version TEXT,
        status TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    """)
    session.shutdown()

@click.command()
@click.option('--name', prompt=True, callback=validate_project_name)
@click.option('--dir', default=".", type=click.Path(exists=True))
@click.option('--git', is_flag=True, help='Initialize git repository')
@click.option('--vscode', is_flag=True, help='Setup VS Code workspace')
@click.pass_context
def init_project(ctx, name, dir, git, vscode):
    """Initialize new RW Agent project with full structure"""

    agents = [
        {'name': 'UI Agent - Web interfaces and dashboards', 'value': 'agent_ui'},
        {'name': 'Storage Agent - Cassandra & Elasticsearch', 'value': 'agent_storage'},
        {'name': 'AI Agent - LLM & RL capabilities', 'value': 'agent_ai'},
        {'name': 'Deployment Agent - CI/CD automation', 'value': 'agent_deployment'},
        {'name': 'Docker Agent - Container management', 'value': 'agent_docker'},
        {'name': 'Kafka Agent - Event streaming', 'value': 'agent_kafka'},
        {'name': 'Redis Agent - Caching & queues', 'value': 'agent_redis'},
        {'name': 'Kubernetes Agent - Orchestration', 'value': 'agent_kubernetes'}
    ]
    required_agents = {'agent_core', 'agent_security'}

    click.echo(f"\n🚀 Initializing RW Agent project: {click.style(name, fg='cyan', bold=True)}")
    click.echo(f"📁 Location: {Path(dir).absolute() / name}")

    selected = questionary.checkbox(
        "Select additional agents to include:",
        choices=agents,
        validate=lambda x: len(x) >= 0
    ).ask()

    if selected is None:
        click.echo("❌ Operation cancelled")
        return

    all_agents = list(required_agents) + selected
    click.echo(f"\n📦 Selected agents: {', '.join(all_agents)}")

    project_path = Path(dir) / name
    _create_project_structure(project_path, all_agents)

    # --- ADD CASSANDRA TABLE CREATION HERE (only if agent_storage is selected) ---
    if 'agent_storage' in all_agents:
        click.echo("🗄️  Creating Cassandra tables for agent metadata ...")
        create_cassandra_tables()
        click.echo("✅ Cassandra tables created.")

    config = {
        'project_id': str(uuid4()),
        'name': name,
        'version': '0.1.0',
        'agents': all_agents,
        'storage': {
            'cassandra': {'enabled': 'agent_storage' in all_agents},
            'elasticsearch': {'enabled': 'agent_storage' in all_agents}
        },
        'ai': {
            'llm': {'enabled': 'agent_ai' in all_agents},
            'rl': {'enabled': 'agent_ai' in all_agents}
        },
        'deployment': {
            'docker': {'enabled': 'agent_docker' in all_agents},
            'kubernetes': {'enabled': 'agent_kubernetes' in all_agents}
        },
        'vscode': {
            'extensions': [
                'ms-python.python',
                'ms-toolsai.jupyter',
                'ms-vscode.vscode-json'
            ]
        }
    }
    (project_path / 'rwagent.json').write_text(json.dumps(config, indent=2))
    _create_env_file(project_path, all_agents)
    _create_requirements_file(project_path, all_agents)
    if vscode:
        _setup_vscode_workspace(project_path)
    if git:
        _init_git_repo(project_path)
    _generate_agent_boilerplate(project_path, all_agents)

    click.echo(f"\n✅ Project {click.style(name, fg='green', bold=True)} created successfully!")
    click.echo(f"📂 Next steps:")
    click.echo(f"   cd {name}")
    click.echo(f"   pip install -r requirements.txt")
    if vscode:
        click.echo(f"   code {name}.code-workspace")

def _create_project_structure(project_path: Path, agents: list):
    """Create the full project directory structure"""
    click.echo("📁 Creating project structure...")
    
    # Base directories
    directories = [
        'src/agents',
        'src/crew/agents',
        'src/crew/tasks',
        'src/utils',
        'src/templates/docker',
        'bin',
        'tests',
        'docs',
        'config'
    ]
    
    # Agent-specific directories
    for agent in agents:
        directories.extend([
            f'src/agents/{agent}',
            f'src/agents/{agent}/tests',
            f'tests/{agent}'
        ])
        
        # Special subdirectories for specific agents
        if agent == 'agent_storage':
            directories.extend([
                f'src/agents/{agent}/cassandra',
                f'src/agents/{agent}/elasticsearch'
            ])
        elif agent == 'agent_ai':
            directories.extend([
                f'src/agents/{agent}/llm',
                f'src/agents/{agent}/rl'
            ])
        elif agent == 'agent_docker':
            directories.append(f'src/agents/{agent}/templates')
    
    # Create all directories
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith('src/'):
            init_file = project_path / directory / '__init__.py'
            if not init_file.exists():
                init_file.touch()

def _create_env_file(project_path: Path, agents: list):
    """Create environment configuration file"""
    env_content = [
        "# RW Agent Project Environment Configuration",
        f"PROJECT_NAME={project_path.name}",
        "ENVIRONMENT=development",
        "",
        "# Core Configuration",
        "LOG_LEVEL=INFO",
        "",
    ]
    
    if 'agent_storage' in agents:
        env_content.extend([
            "# Storage Configuration",
            "CASSANDRA_HOST=localhost",
            "CASSANDRA_PORT=9042",
            "CASSANDRA_KEYSPACE=rwagent",
            "CASSANDRA_USER=",
            "CASSANDRA_PASS=",
            "",
            "ELASTICSEARCH_URL=http://localhost:9200",
            "ELASTICSEARCH_USER=",
            "ELASTICSEARCH_PASS=",
            "",
        ])
    
    if 'agent_ai' in agents:
        env_content.extend([
            "# AI Configuration",
            "ANTHROPIC_API_KEY=",
            "GROQ_API_KEY=",
            "OPENAI_API_KEY=",
            "",
        ])
    
    if 'agent_kafka' in agents:
        env_content.extend([
            "# Kafka Configuration",
            "KAFKA_BOOTSTRAP_SERVERS=localhost:9092",
            "",
        ])
    
    if 'agent_redis' in agents:
        env_content.extend([
            "# Redis Configuration",
            "REDIS_URL=redis://localhost:6379",
            "",
        ])
    
    (project_path / '.env').write_text('\n'.join(env_content))

def _create_requirements_file(project_path: Path, agents: list):
    """Create requirements.txt based on selected agents"""
    requirements = [
        "# Core dependencies",
        "click>=8.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "",
    ]
    
    if 'agent_storage' in agents:
        requirements.extend([
            "# Storage dependencies",
            "cassandra-driver>=3.28.0",
            "elasticsearch>=8.11.0",
            "",
        ])
    
    if 'agent_ai' in agents:
        requirements.extend([
            "# AI dependencies",
            "anthropic>=0.3.0",
            "groq>=0.4.0",
            "gymnasium>=0.29.0",  # For RL
            "stable-baselines3>=2.0.0",  # For RL
            "",
        ])
    
    if 'agent_docker' in agents:
        requirements.extend([
            "# Docker dependencies",
            "docker>=6.0.0",
            "",
        ])
    
    if 'agent_kafka' in agents:
        requirements.extend([
            "# Kafka dependencies",
            "confluent-kafka>=2.0.0",
            "",
        ])
    
    if 'agent_redis' in agents:
        requirements.extend([
            "# Redis dependencies",
            "redis>=5.0.0",
            "",
        ])
    
    if 'agent_kubernetes' in agents:
        requirements.extend([
            "# Kubernetes dependencies",
            "kubernetes>=28.0.0",
            "",
        ])
    
    (project_path / 'requirements.txt').write_text('\n'.join(requirements))

def _setup_vscode_workspace(project_path: Path):
    """Setup VS Code workspace with optimal settings"""
    # Create .vscode directory
    vscode_dir = project_path / '.vscode'
    vscode_dir.mkdir(exist_ok=True)
    
    # VS Code settings
    settings = {
        "python.analysis.typeCheckingMode": "basic",
        "python.formatting.provider": "black",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            ".env": False
        },
        "search.exclude": {
            "**/node_modules": True,
            "**/dist": True
        }
    }
    
    (vscode_dir / 'settings.json').write_text(json.dumps(settings, indent=2))
    
    # Create workspace file
    workspace = {
        "folders": [
            {"path": "."}
        ],
        "settings": settings,
        "extensions": {
            "recommendations": [
                "ms-python.python",
                "ms-toolsai.jupyter",
                "ms-vscode.vscode-json",
                "charliermarsh.ruff"
            ]
        }
    }
    
    (project_path / f'{project_path.name}.code-workspace').write_text(
        json.dumps(workspace, indent=2)
    )

def _init_git_repo(project_path: Path):
    """Initialize git repository with proper .gitignore"""
    click.echo("🔧 Initializing git repository...")
    
    try:
        subprocess.run(['git', 'init'], cwd=project_path, check=True, capture_output=True)
        
        # Create .gitignore
        gitignore_content = [
            "# Python",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "env/",
            "venv/",
            ".venv/",
            "",
            "# Environment files",
            ".env",
            ".env.local",
            "",
            "# IDE",
            ".vscode/",
            ".idea/",
            "",
            "# Logs",
            "*.log",
            "logs/",
            "",
            "# Database",
            "*.db",
            "*.sqlite3",
            "",
            "# OS",
            ".DS_Store",
            "Thumbs.db",
        ]
        
        (project_path / '.gitignore').write_text('\n'.join(gitignore_content))
        
        # Initial commit
        subprocess.run(['git', 'add', '.'], cwd=project_path, check=True)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial RW Agent project setup'],
            cwd=project_path, 
            check=True,
            capture_output=True
        )
        
        click.echo("✅ Git repository initialized")
        
    except subprocess.CalledProcessError:
        click.echo("⚠️  Git initialization failed (git might not be installed)")

def _generate_agent_boilerplate(project_path: Path, agents: list):
    """Generate basic boilerplate for each agent"""
    click.echo("🛠️  Generating agent boilerplate...")
    
    for agent in agents:
        agent_path = project_path / 'src' / 'agents' / agent
        
        # Create core.py for each agent
        core_content = f'''"""
{agent.replace('_', ' ').title()} implementation.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class {agent.replace('_', '').title()}:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize()
    
    def initialize(self):
        """Initialize {agent} components"""
        logger.info(f"Initializing {agent}")
        # TODO: Add initialization logic
    
    def validate_config(self) -> bool:
        """Validate {agent} configuration"""
        # TODO: Add validation logic
        return True
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute {agent} operations"""
        # TODO: Add execution logic
        raise NotImplementedError(f"{agent} execution not implemented")
'''
        
        (agent_path / 'core.py').write_text(core_content)
        
        # Create README for each agent
        readme_content = f'''# {agent.replace('_', ' ').title()}

## Overview
This agent handles {agent.replace('_', ' ')} functionality.

## Configuration
TODO: Document configuration options

## Usage
TODO: Document usage examples

## API
TODO: Document API endpoints
'''
        
        (agent_path / 'README.md').write_text(readme_content)

