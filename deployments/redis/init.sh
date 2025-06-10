#!/bin/bash
set -eo pipefail

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REDIS_CONTAINER="redis"
COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"
ENV_FILE="$(dirname "$0")/../../.env"
MAX_WAIT=300  # 5 minutes timeout
DEFAULT_PORT=6379

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------
load_env() {
    # Create .env if missing
    if [ ! -f "$ENV_FILE" ]; then
        echo "Creating .env from sample..."
        cp "$(dirname "$0")/../../.env_sample" "$ENV_FILE"
    fi

    # Set defaults
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
    sed -i.bak "s/- \"[0-9]\+:6379\"/- \"${new_port}:6379\"/" "$COMPOSE_FILE"
    rm -f "${ENV_FILE}.bak" "${COMPOSE_FILE}.bak"
}

# --------------------------------------------------
# Docker Container Management
# --------------------------------------------------
manage_container() {
    # Check port conflict
    if nc -z "$REDIS_HOST" "$REDIS_PORT"; then
        echo "Port $REDIS_PORT in use, finding alternative..."
        NEW_PORT=$(find_available_port $REDIS_PORT)
        update_port_config "$NEW_PORT"
        export REDIS_PORT=$NEW_PORT
    fi

    # Start/recreate container
    if docker ps -a --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$REDIS_CONTAINER")" != "true" ]; then
            echo "Starting existing Redis container..."
            docker start "$REDIS_CONTAINER"
        fi
    else
        echo "Creating new Redis container..."
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
}

# --------------------------------------------------
# Service Readiness Check
# --------------------------------------------------
wait_for_redis() {
    echo "Waiting for Redis to initialize..."
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
    # Load environment with fallbacks
    load_env

    # Manage container with port handling
    manage_container
    wait_for_redis

    echo "✅ Redis initialized successfully"
    echo "    Host: redis://${REDIS_HOST}:${REDIS_PORT}"
}

# Start initialization
main

