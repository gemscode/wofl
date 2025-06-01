#!/bin/bash

# Create base directory structure
mkdir -p rw_agent/{bin,src/{agents/{agent_core,agent_security,agent_ui,agent_deployment,agent_kafka,agent_redis,agent_storage,agent_kubernetes,agent_docker},crew/{agents,tasks},utils,templates/docker},docs}

# Create root files
touch rw_agent/requirements.txt
touch rw_agent/README.md
touch rw_agent/.env
touch rw_agent/.gitignore

# Create bin files
touch rw_agent/bin/init_project.py
touch rw_agent/bin/create_cassandra_tables.cql
chmod +x rw_agent/bin/init_project.py

# Create src files
## Agents
for agent in core security ui deployment kafka redis storage kubernetes docker; do
    mkdir -p "rw_agent/src/agents/agent_${agent}/tests"
    touch "rw_agent/src/agents/agent_${agent}/__init__.py"
    touch "rw_agent/src/agents/agent_${agent}/core.py"
done

## Crew
touch rw_agent/src/crew/__init__.py
touch rw_agent/src/crew/orchestrator.py
mkdir -p rw_agent/src/crew/agents
touch rw_agent/src/crew/agents/__init__.py
mkdir -p rw_agent/src/crew/tasks
touch rw_agent/src/crew/tasks/__init__.py

## Utils
touch rw_agent/src/utils/__init__.py
touch rw_agent/src/utils/cassandra_manager.py
touch rw_agent/src/utils/git_manager.py
touch rw_agent/src/utils/docker_manager.py
touch rw_agent/src/utils/config_loader.py

## Templates
touch rw_agent/src/templates/docker/Dockerfile.j2

# Populate basic file contents

# .env file
cat > rw_agent/.env << EOL
# Project Configuration
PROJECT_NAME=rw_agent
ENVIRONMENT=development

# Database Settings
CASSANDRA_HOST=localhost
CASSANDRA_PORT=9042
CASSANDRA_KEYSPACE=rw_agent
CASSANDRA_USER=
CASSANDRA_PASS=

# Git Configuration
GIT_BASE_PATH=./repos
GITHUB_TOKEN=

# Docker Settings
DOCKER_REGISTRY=registry.rwagent.com
PYTHON_VERSION=3.9
EOL

# .gitignore
cat > rw_agent/.gitignore << EOL
# Environment files
.env
.env.local

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Database
*.db
*.dump

# Docker
docker-compose.override.yml

# IDE
.vscode/
.idea/
EOL

# requirements.txt
cat > rw_agent/requirements.txt << EOL
cassandra-driver==3.28.0
gitpython==3.1.40
python-dotenv==1.1.0
jinja2==3.1.2
crewai==0.28.8
docker==7.0.0
python-baseconv==1.2.2
EOL

# Basic README
cat > rw_agent/README.md << EOL
# RW Agent Framework

A scalable agent framework with modular components.

## Getting Started

1. Set up Cassandra database
2. Configure environment variables
3. Initialize project using bin/init_project.py

## Core Components

- Agent Orchestration
- Automated Deployment
- Version Control Integration
EOL

# Create basic agent boilerplate
for agent in core security ui deployment kafka redis storage kubernetes docker; do
    cat > "rw_agent/src/agents/agent_${agent}/core.py" << EOL
class ${agent^}Agent:
    def __init__(self, config):
        self.config = config
        self.initialize()
    
    def initialize(self):
        """Initialize agent components"""
        pass
    
    def validate_config(self):
        """Validate agent-specific configuration"""
        return True
    
    def execute(self, task):
        """Main execution method"""
        raise NotImplementedError("Agent execution not implemented")
EOL
done

# Create basic Docker template
cat > rw_agent/src/templates/docker/Dockerfile.j2 << EOL
# Auto-generated Dockerfile for {{ project_id }}
FROM python:{{ python_version }}-slim

WORKDIR /app

# Install base dependencies
RUN pip install --no-cache-dir \\
    cassandra-driver \\
    gitpython \\
    jinja2 \\
    crewai

{% if 'agent_kafka' in agents %}
# Kafka dependencies
RUN pip install confluent-kafka
{% endif %}

{% if 'agent_redis' in agents %}
# Redis dependencies
RUN pip install redis
{% endif %}

# Copy project files
COPY . .

CMD ["python", "main.py"]
EOL

echo "Project structure created successfully!"

