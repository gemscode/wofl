#!/bin/bash
set -eo pipefail

# --------------------------------------------------
# Configuration (packaging-compatible paths)
# --------------------------------------------------
if [ -n "$MEIPASS" ]; then
    # Packaged mode (PyInstaller)
    ROOT_DIR="$MEIPASS"
else 
    # Development mode
    ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
fi

REDIS_CONTAINER="redis"
COMPOSE_FILE="$ROOT_DIR/src/deployments/redis/docker-compose.yml"
ENV_FILE="$(pwd)/.env"
MAX_WAIT=300
DEFAULT_PORT=6379

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------
load_env() {
    # Create .env in current directory if missing
    if [ ! -f "$ENV_FILE" ]; then
        echo "📄 Creating .env from packaged template..."
        cp "$ROOT_DIR/.env_sample" "$ENV_FILE"
    fi

    # Source and set defaults
    source "$ENV_FILE"
    export REDIS_HOST=${REDIS_HOST:-localhost}
    export REDIS_PORT=${REDIS_PORT:-$DEFAULT_PORT}

    # Update .env with current values
    grep -q "^REDIS_PORT=" "$ENV_FILE" || echo "REDIS_PORT=$REDIS_PORT" >> "$ENV_FILE"
}

# --------------------------------------------------
# Port Management
# --------------------------------------------------
find_available_port() {
    local base_port=$1
    while true; do
        if ! nc -z "$REDIS_HOST" "$base_port"; then
            echo "$base_port"
            return
        fi
        ((base_port++))
    done
}

update_port_config() {
    local new_port=$1
    sed -i.bak "s/^REDIS_PORT=.*/REDIS_PORT=$new_port/" "$ENV_FILE"
    rm -f "${ENV_FILE}.bak"
}

# --------------------------------------------------
# Docker Container Management
# --------------------------------------------------
manage_container() {
    # Handle port conflicts
    if nc -z "$REDIS_HOST" "$REDIS_PORT"; then
        echo "🚨 Port $REDIS_PORT in use, finding alternative..."
        NEW_PORT=$(find_available_port $REDIS_PORT)
        update_port_config "$NEW_PORT"
        export REDIS_PORT=$NEW_PORT
    fi

    # Manage container state
    if docker ps -a --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$REDIS_CONTAINER")" = "false" ]; then
            echo "🚀 Starting existing Redis container..."
            docker start "$REDIS_CONTAINER"
        else
            echo "✅ Redis already running"
        fi
    else
        echo "🌟 Creating new Redis container..."
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
}

# --------------------------------------------------
# Service Readiness Check
# --------------------------------------------------
wait_for_redis() {
    echo "⏳ Waiting for Redis initialization..."
    local counter=0
    until docker exec "$REDIS_CONTAINER" redis-cli ping | grep -q "PONG"; do
        sleep 5
        counter=$((counter + 5))
        if [ $counter -ge $MAX_WAIT ]; then
            echo "❌ Redis initialization timed out"
            exit 1
        fi
    done
}

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
main() {
    echo "🚀 Starting Redis initialization..."
    load_env
    manage_container
    wait_for_redis

    echo ""
    echo "🎉 Redis ready!"
    echo "   Connection URL: redis://${REDIS_HOST}:${REDIS_PORT}"
    echo "   Try: redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT}"
}

main "$@"

