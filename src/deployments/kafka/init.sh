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

KAFKA_CONTAINER="wolfx0-kafka"
SETUP_CONTAINER="wolfx0-kafka-setup"
COMPOSE_FILE="$ROOT_DIR/src/deployments/kafka/docker-compose.yml"
ENV_FILE="$(pwd)/.env"
MAX_WAIT=300
DEFAULT_CLIENT_PORT=9092
DEFAULT_CONTROLLER_PORT=9093

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------
load_env() {
    # Create .env in current directory if missing
    if [ ! -f "$ENV_FILE" ]; then
        echo "📄 Creating .env from packaged template..."
        cp "$ROOT_DIR/.env_sample" "$ENV_FILE"
    fi

    # Source environment with defaults
    source "$ENV_FILE"
    export KAFKA_HOST=${KAFKA_HOST:-localhost}
    export KAFKA_CLIENT_PORT=${KAFKA_CLIENT_PORT:-$DEFAULT_CLIENT_PORT}
    export KAFKA_CONTROLLER_PORT=${KAFKA_CONTROLLER_PORT:-$DEFAULT_CONTROLLER_PORT}

    # Update .env with current values
    grep -q "^KAFKA_CLIENT_PORT=" "$ENV_FILE" || echo "KAFKA_CLIENT_PORT=$KAFKA_CLIENT_PORT" >> "$ENV_FILE"
    grep -q "^KAFKA_CONTROLLER_PORT=" "$ENV_FILE" || echo "KAFKA_CONTROLLER_PORT=$KAFKA_CONTROLLER_PORT" >> "$ENV_FILE"
}

# --------------------------------------------------
# Port Management
# --------------------------------------------------
find_available_port() {
    local base_port=$1
    while true; do
        if ! nc -z "$KAFKA_HOST" "$base_port"; then
            echo "$base_port"
            return
        fi
        ((base_port++))
    done
}

update_port_config() {
    local new_client_port=$1
    local new_controller_port=$2
    
    # Update .env in current directory
    sed -i.bak \
        -e "s/^KAFKA_CLIENT_PORT=.*/KAFKA_CLIENT_PORT=$new_client_port/" \
        -e "s/^KAFKA_CONTROLLER_PORT=.*/KAFKA_CONTROLLER_PORT=$new_controller_port/" \
        "$ENV_FILE"
    
    # Update packaged compose file reference
    sed -i.bak \
        -e "s/- \"[0-9]\+:9092\"/- \"${new_client_port}:9092\"/" \
        -e "s/- \"[0-9]\+:9093\"/- \"${new_controller_port}:9093\"/" \
        "$COMPOSE_FILE"
    
    rm -f "${ENV_FILE}.bak" "${COMPOSE_FILE}.bak"
}

# --------------------------------------------------
# Docker Container Management
# --------------------------------------------------
manage_container() {
    # Check port conflicts
    if nc -z "$KAFKA_HOST" "$KAFKA_CLIENT_PORT" || 
       nc -z "$KAFKA_HOST" "$KAFKA_CONTROLLER_PORT"; then
        echo "Port conflict detected, finding alternatives..."
        NEW_CLIENT_PORT=$(find_available_port $KAFKA_CLIENT_PORT)
        NEW_CONTROLLER_PORT=$(find_available_port $KAFKA_CONTROLLER_PORT)
        update_port_config "$NEW_CLIENT_PORT" "$NEW_CONTROLLER_PORT"
        export KAFKA_CLIENT_PORT=$NEW_CLIENT_PORT
        export KAFKA_CONTROLLER_PORT=$NEW_CONTROLLER_PORT
    fi

    # Start/recreate container using packaged compose file
    if docker ps -a --format '{{.Names}}' | grep -q "^${KAFKA_CONTAINER}$"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$KAFKA_CONTAINER")" != "true" ]; then
            echo "Starting existing Kafka container..."
            docker start "$KAFKA_CONTAINER"
        else
            echo "Kafka already running"
        fi
    else
        echo "Creating new Kafka container..."
        docker-compose -f "$COMPOSE_FILE" up -d kafka
    fi
}

# --------------------------------------------------
# Service Readiness Check
# --------------------------------------------------
wait_for_kafka() {
    echo "Waiting for Kafka to initialize..."
    local counter=0
    until docker exec "$KAFKA_CONTAINER" kafka-topics.sh --bootstrap-server localhost:9092 --list &>/dev/null; do
        docker logs "$SETUP_CONTAINER" 2>&1 | tail -n 100 || true
        sleep 10
        counter=$((counter + 10))
        if [ $counter -ge $MAX_WAIT ]; then
            echo "❌ Kafka initialization timed out"
            exit 1
        fi
    done
}

# --------------------------------------------------
# Topic Setup
# --------------------------------------------------
run_setup() {
    echo "Running Kafka setup..."
    docker-compose -f "$COMPOSE_FILE" up -d setup

    echo "Waiting for setup completion..."
    local counter=0
    local max_wait=60
    local interval=3

    while true; do
        logs=$(docker logs "$SETUP_CONTAINER" 2>&1 || true)

        if echo "$logs" | grep -q "Successfully produced sample messages"; then
            echo "✅ Kafka setup complete"
            return
        fi

        if ! docker ps -a --format '{{.Names}}' | grep -q "^$SETUP_CONTAINER$"; then
            echo "❌ Kafka setup container vanished unexpectedly"
            echo "$logs"
            exit 1
        fi

        if [ $counter -ge $max_wait ]; then
            echo "❌ Kafka setup timed out"
            echo "$logs"
            exit 1
        fi

        sleep $interval
        counter=$((counter + interval))
    done
}

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
main() {
    load_env
    manage_container
    wait_for_kafka
    run_setup

    echo "✅ Kafka initialized successfully"
    echo "    Client: ${KAFKA_HOST}:${KAFKA_CLIENT_PORT}"
    echo "    Controller: ${KAFKA_HOST}:${KAFKA_CONTROLLER_PORT}"
}

main "$@"

