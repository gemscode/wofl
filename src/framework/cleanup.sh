#!/bin/bash
set -eo pipefail

# -------------------------------
# Resolve project root
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
# Remove virtual environment
# -------------------------------
cleanup_venv() {
    if [ -d "$ROOT_DIR/framework/core_env" ]; then
        echo "🧹 Removing virtual environment..."
        rm -rf "$ROOT_DIR/framework/core_env"
    fi
}

# -------------------------------
# Uninstall rwagent CLI
# -------------------------------
uninstall_cli() {
    if command -v rwagent &>/dev/null; then
        echo "🧹 Uninstalling global rwagent CLI..."
        pip uninstall -y rwagent questionary || true
    fi
}

# -------------------------------
# Remove compiled requirements
# -------------------------------
remove_requirements_artifacts() {
    echo "🧹 Removing requirements artifacts..."
    rm -f "$ROOT_DIR/requirements.txt"
    rm -f "$ROOT_DIR/.requirements.sha256"
}

# -------------------------------
# Stop and remove Docker containers
# -------------------------------
cleanup_services() {
    echo "🛑 Stopping related Docker containers..."

    for service in cassandra elasticsearch kafka redis; do
        container=$(docker ps -aqf "name=wolfx0-$service")
        if [ -n "$container" ]; then
            echo "🗑 Removing container for $service..."
            docker stop "$container" || true
            docker rm "$container" || true
        fi
    done
}

# -------------------------------
# Remove generated metadata
# -------------------------------
remove_metadata_files() {
    echo "🧹 Removing generated metadata..."
    rm -f "$ROOT_DIR/.env"
    rm -f "$ROOT_DIR/framework/rwagent.json"
}

# -------------------------------
# Entry point
# -------------------------------
main() {
    echo "⚠️  This will clean up your framework environment..."
    read -rp "Are you sure? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "❌ Aborted."
        exit 1
    fi

    cleanup_venv
    uninstall_cli
    remove_requirements_artifacts
    cleanup_services
    remove_metadata_files

    echo "✅ Cleanup complete."
}

main "$@"

