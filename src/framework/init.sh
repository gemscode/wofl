#!/bin/zsh
set -eo pipefail

# -------------------------------
# Universal path resolution
# -------------------------------
if [ -n "$MEIPASS" ]; then
    # Packaged mode (PyInstaller) - MEIPASS set by launcher.py
    ROOT_DIR="$MEIPASS"
else 
    # Development mode
    ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# -------------------------------
# Environment configuration
# -------------------------------
init_env() {
    local env_sample="$ROOT_DIR/.env_sample"
    local env_file="$(pwd)/.env"  # Always create in current directory

    if [ ! -f "$env_file" ]; then
        if [ -f "$env_sample" ]; then
            echo "📄 Creating .env from .env_sample..."
            cp "$env_sample" "$env_file"
        else
            echo "❌ Critical: Missing .env_sample in package"
            exit 1
        fi
    fi

    # Update existing .env with new variables
    echo "🔍 Updating environment variables..."
    grep -E '^[A-Z_]+=' "$env_sample" | while read -r line; do
        var_name="${line%%=*}"
        if ! grep -q "^${var_name}=" "$env_file"; then
            echo "$line" >> "$env_file"
        fi
    done

    # Load environment from current directory
    set -a
    source "$env_file"
    set +a
}

# -------------------------------
# Core environment activation
# -------------------------------
activate_core_env() {
    if [ -z "$MEIPASS" ]; then  # Development mode
        local venv_path="$ROOT_DIR/core_env"

        echo "⚙️  Activating core environment..."
        if [ -f "$venv_path/bin/activate" ]; then
            source "$venv_path/bin/activate"

            # Set library paths for Cassandra
            export DYLD_FALLBACK_LIBRARY_PATH="$ROOT_DIR/lib/libev-install/lib:$DYLD_FALLBACK_LIBRARY_PATH"
            export C_INCLUDE_PATH="$ROOT_DIR/lib/libev-install/include:$C_INCLUDE_PATH"

            echo "✅ Environment ready"
        else
            echo "❌ Critical: Missing core environment"
            exit 1
        fi
    else  # Packaged mode
        echo "📦 Packaged mode: Using embedded Python"
        # Add packaged Python to PATH
        export PATH="$ROOT_DIR/bin:$PATH"
    fi
}


# -------------------------------
# Service management
# -------------------------------
init_services() {
    echo "🚀 Initializing infrastructure..."
    local services=("cassandra" "elasticsearch" "kafka" "redis" "k3s")

    for service in "${services[@]}"; do
        local service_script="$ROOT_DIR/src/deployments/$service/init.sh"
        
        if [ -x "$service_script" ]; then
            echo "🔧 Starting $service..."
            (cd "$(dirname "$service_script")" && ./"$(basename "$service_script")")
        else
            echo "⚠️  Missing service script: $service_script"
        fi
    done
}

# -------------------------------
# Database setup
# -------------------------------
init_schemas() {
    # Cassandra schema
    local cql_file="$ROOT_DIR/src/deployments/cassandra/create_tables.cql"
    if [ -f "$cql_file" ]; then
        echo "🔨 Applying Cassandra schema..."
        if docker ps | grep -q wolfx0-cassandra; then
            docker cp "$cql_file" wolfx0-cassandra:/create_tables.cql
            docker exec wolfx0-cassandra cqlsh -f /create_tables.cql || true
            sed -i.bak "s/^INSTALLED_CASSANDRA=.*/INSTALLED_CASSANDRA=true/" "$(pwd)/.env"
        else
            echo "⚠️  Cassandra container not running"
        fi
    fi

    # Elasticsearch index
    if [ -n "$ELASTICSEARCH_HOST" ] && [ -n "$ELASTICSEARCH_HTTP_PORT" ]; then
        echo "📈 Creating Elasticsearch index..."
        curl -X PUT "http://${ELASTICSEARCH_HOST}:${ELASTICSEARCH_HTTP_PORT}/agent_registry" \
            -H 'Content-Type: application/json' -d'
        {
            "mappings": {
                "properties": {
                    "agent_id": { "type": "keyword" },
                    "name": { "type": "text" },
                    "description": { "type": "text" },
                    "created_at": { "type": "date" }
                }
            }
        }' || true
    fi
}

# -------------------------------
# Framework registration
# -------------------------------
register_framework() {
    echo "📦 Registering framework agent..."
    local register_script="$ROOT_DIR/src/framework/rw_agent/bin/register_agent.py"
    
    if ! python "$register_script"; then
        echo "❌ Critical: Framework registration failed"
        exit 1
    fi
    echo "✅ Framework agent active"
}

# -------------------------------
# CLI installation
# -------------------------------
setup_cli() {
    echo "🛠  Configuring CLI..."
    if ! command -v rwagent &>/dev/null; then
        echo "📦 Installing CLI tools..."
        pip install -e "$ROOT_DIR/src/framework"
        echo "✅ CLI installed"
    else
        echo "✅ CLI already available"
    fi
}

# -------------------------------
# Main workflow
# -------------------------------
main() {
    echo "🔨 Starting WolfX initialization..."
    init_env
    activate_core_env
    init_services
    init_schemas
    register_framework
    setup_cli

    echo "\n🎉 WolfX initialization complete!"
    echo "➡️  Available commands:"
    echo "   rwagent init-project <dir>      Create new project"
    echo "   rwagent deploy-service <name>   Deploy a service"
    echo "   rwagent config <key> <value>    Update configuration"
    echo "   rwagent --help                  Show all commands"
}

main "$@"
