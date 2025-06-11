#!/bin/bash
set -eo pipefail

# -------------------------------
# Find project root
# -------------------------------
find_root() {
    if command -v git &>/dev/null && git rev-parse --show-toplevel &>/dev/null; then
        git rev-parse --show-toplevel
    else
        cd "$(dirname "$0")/.." && pwd
    fi
}

ROOT_DIR="$(find_root)"
cd "$ROOT_DIR"

# -------------------------------
# Load & patch .env from .env_sample
# -------------------------------
init_env() {
    if [ ! -f "$ROOT_DIR/.env" ]; then
        if [ -f "$ROOT_DIR/.env_sample" ]; then
            echo "Creating .env from .env_sample..."
            cp "$ROOT_DIR/.env_sample" "$ROOT_DIR/.env"
        else
            echo "❌ .env_sample not found. Cannot create .env."
            exit 1
        fi
    else
        while IFS= read -r line; do
            if [[ "$line" =~ ^[A-Z_] && ! "$line" =~ ^# ]]; then
                var_name="${line%%=*}"
                grep -q "^${var_name}=" "$ROOT_DIR/.env" || echo "$line" >> "$ROOT_DIR/.env"
            fi
        done < "$ROOT_DIR/.env_sample"
    fi
    set -a
    source "$ROOT_DIR/.env"
    set +a
}

# -------------------------------
# Virtualenv setup (with activation)
# -------------------------------
setup_core_venv() {
    local req_in="$ROOT_DIR/core_requirements.in"
    local req_txt="$ROOT_DIR/core_requirements.txt"
    local checksum_file="$ROOT_DIR/.core_requirements.sha256"

    if [ ! -d "$ROOT_DIR/framework/core_env" ]; then
        echo "Creating core virtual environment (core_env)..."
        python3 -m venv "$ROOT_DIR/framework/core_env"
    fi

    source "$ROOT_DIR/framework/core_env/bin/activate"

    if ! pip show pip-tools &>/dev/null; then
        pip install pip-tools
    fi

    if [ ! -f "$req_in" ]; then
        echo "❌ $req_in not found."
        exit 1
    fi

    local current_checksum
    current_checksum=$(sha256sum "$req_in" | awk '{print $1}')

    local stored_checksum=""
    [ -f "$checksum_file" ] && stored_checksum=$(cat "$checksum_file")

    if [ "$current_checksum" != "$stored_checksum" ]; then
        echo "🔄 Detected change in $req_in. Rebuilding dependencies..."
        pip-compile "$req_in" -o "$req_txt"
        pip install -r "$req_txt"
        echo "$current_checksum" > "$checksum_file"
    else
        echo "✅ core_requirements.in unchanged. Skipping dependency rebuild."
    fi

    export PYTHONPATH="$ROOT_DIR/framework:$PYTHONPATH"
    if ! grep -q 'PYTHONPATH=.*framework' "$ROOT_DIR/framework/core_env/bin/activate"; then
        echo "export PYTHONPATH=\"$ROOT_DIR/framework:\$PYTHONPATH\"" >> "$ROOT_DIR/framework/core_env/bin/activate"
    fi
}

# -------------------------------
# Initialize each service
# -------------------------------
init_services() {
    for service in cassandra elasticsearch kafka redis k3s; do
        echo "🔧 Initializing $service..."
        if [ -x "$ROOT_DIR/deployments/$service/init.sh" ]; then
            (cd "$ROOT_DIR/deployments/$service" && ./init.sh)
        else
            echo "⚠️  $ROOT_DIR/deployments/$service/init.sh not found or not executable"
        fi
    done
}

# -------------------------------
# Cassandra schema creation
# -------------------------------
init_schemas() {
    if [ -f "$ROOT_DIR/deployments/cassandra/create_tables.cql" ]; then
        echo "Initializing Cassandra schema..."

        CASS_CONTAINER=$(docker ps --format '{{.Names}}' | grep wolfx0-cassandra || true)
        if [ -n "$CASS_CONTAINER" ]; then
            docker cp "$ROOT_DIR/deployments/cassandra/create_tables.cql" "$CASS_CONTAINER":/var/lib/cassandra/create_tables.cql || true
            docker exec "$CASS_CONTAINER" cqlsh -f /var/lib/cassandra/create_tables.cql || true

            sed -i.bak "s/^INSTALLED_CASSANDRA=.*/INSTALLED_CASSANDRA=true/" "$ROOT_DIR/.env"
            rm -f "$ROOT_DIR/.env.bak"
        else
            echo "⚠️  Could not find Cassandra container to load schema."
        fi
    fi

    if [ -n "$ELASTICSEARCH_HOST" ] && [ -n "$ELASTICSEARCH_HTTP_PORT" ]; then
        echo "Ensuring Elasticsearch index exists..."
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
# Register main framework agent
# -------------------------------
register_framework() {
    echo "📦 Registering framework..."

    echo "🚀 Running register_agent.py..."
    if ! python "$ROOT_DIR/framework/rw_agent/bin/register_agent.py"; then
        echo "❌ Failed to register framework agent"
        exit 1
    fi

    echo "✅ Framework agent registered successfully"
}

# -------------------------------
# Install RW CLI globally (editable mode)
# -------------------------------
install_cli() {
    echo "🛠 Checking rwagent CLI installation..."

    if ! which rwagent &>/dev/null; then
        echo "📦 Installing rwagent CLI in editable mode..."
        pip install -e "$ROOT_DIR/framework"
        pip install questionary
    else
        echo "✅ rwagent CLI already available in PATH. Skipping reinstall."
    fi

    echo "🔍 Verifying rwagent CLI availability..."
    if ! command -v rwagent &>/dev/null; then
        echo "❌ 'rwagent' not found in PATH. Try running 'source framework/core_env/bin/activate' manually."
    else
        echo "✅ 'rwagent' is now available."
    fi
}

# -------------------------------
# Ensure metadata exists
# -------------------------------
ensure_metadata_json() {
    echo "📁 Ensuring rwagent.json exists in framework..."
    local FRAMEWORK_JSON="$ROOT_DIR/framework/rwagent.json"

    if [ ! -f "$FRAMEWORK_JSON" ]; then
        echo "🔧 rwagent.json not found. Generating from Cassandra metadata..."
        if ! python "$ROOT_DIR/framework/utils/create_metadata.py"; then
            echo "❌ Failed to generate rwagent.json"
            exit 1
        fi
        echo "✅ Created rwagent.json at $FRAMEWORK_JSON"
    else
        echo "✅ rwagent.json already exists."
    fi
}

# -------------------------------
# Entry point
# -------------------------------
main() {
    init_env
    setup_core_venv
    init_services
    init_schemas
    register_framework
    install_cli
    ensure_metadata_json

    echo "✅ Framework initialization complete"
    echo "➡️  You can now use the CLI to manage the framework (inside core_env):"
    echo "   • rwagent init-project <dir>         # Create a new agent-based project"
    echo "   • rwagent deploy-service <name>      # Deploy or update a service"
    echo "   • rwagent config <key> <value>       # Update CLI configuration"
    echo "   • rwagent integrity                  # Run integrity checks"
    echo "   • rwagent list-agents                # Show all available agents"
    echo "   • rwagent status <project_dir>       # Show status of a specific project"
    echo "   • rwagent --help                     # View all available commands"
}

main "$@"

