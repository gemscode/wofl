#!/bin/bash
set -eo pipefail

# --------------------------------------------
# Configuration
# --------------------------------------------
PORT_RANGE_START=9042
PORT_RANGE_END=9100
CASSANDRA_CONTAINER="wolfx0-cassandra"
VOLUME_NAME="cassandra_data"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"
CQL_FILE="$(dirname "$0")/create_tables.cql"

# --------------------------------------------
# Load .env or initialize it
# --------------------------------------------
load_env() {
    [ -f "$ENV_FILE" ] || cp "$ROOT_DIR/.env_sample" "$ENV_FILE"
    source "$ENV_FILE"

    : ${CASSANDRA_HOST:=localhost}
    : ${CASSANDRA_PORT:=9042}
    : ${INSTALLED_CASSANDRA:=false}
    : ${INSTALLED_CASSANDRA_PORT:=$CASSANDRA_PORT}
}

# --------------------------------------------
# Find available port in range
# --------------------------------------------
find_available_port() {
    local base_port=$1
    while [ $base_port -le $PORT_RANGE_END ]; do
        if ! lsof -i :$base_port >/dev/null && ! docker ps --format "{{.Ports}}" | grep -q ":$base_port->"; then
            echo "$base_port"
            return
        fi
        ((base_port++))
    done
    echo "ERROR: No ports available in $PORT_RANGE_START-$PORT_RANGE_END" >&2
    exit 1
}

# --------------------------------------------
# Update .env and docker-compose.yml
# --------------------------------------------
update_configs() {
    local new_port=$1
    sed -i.bak \
        -e "s/^CASSANDRA_PORT=.*/CASSANDRA_PORT=$new_port/" \
        -e "s/^INSTALLED_CASSANDRA_PORT=.*/INSTALLED_CASSANDRA_PORT=$new_port/" \
        "$ENV_FILE"

    sed -i.bak \
        -E "s/- \"[0-9]+:9042\"/- \"$new_port:9042\"/" \
        "$COMPOSE_FILE"

    rm -f "$ENV_FILE.bak" "$COMPOSE_FILE.bak"
}

# --------------------------------------------
# Start Cassandra container
# --------------------------------------------
start_container() {
    echo "Removing old Cassandra container (if any)..."
    docker rm -f "$CASSANDRA_CONTAINER" 2>/dev/null || true
    docker volume rm -f "$VOLUME_NAME" 2>/dev/null || true

    echo "Starting Cassandra on port $CASSANDRA_PORT..."
    docker-compose -f "$COMPOSE_FILE" up -d
}

# --------------------------------------------
# Wait for Cassandra readiness
# --------------------------------------------
wait_for_cassandra() {
    echo "Waiting for Cassandra to initialize..."
    local attempts=0
    local max_attempts=30
    local delay=5

    until docker exec "$CASSANDRA_CONTAINER" cqlsh -e "DESCRIBE KEYSPACES" &>/dev/null; do
        sleep $delay
        ((attempts++))
        if [ $attempts -ge $max_attempts ]; then
            echo "❌ Cassandra did not initialize in time"
            docker logs "$CASSANDRA_CONTAINER"
            exit 1
        fi
    done
    echo "✅ Cassandra is ready."
}

# --------------------------------------------
# Main
# --------------------------------------------
main() {
    load_env

    if [ "$INSTALLED_CASSANDRA" = "true" ]; then
        echo "✅ Cassandra already installed on port $CASSANDRA_PORT"
        return
    fi

    if lsof -i :$CASSANDRA_PORT || docker ps --format "{{.Ports}}" | grep -q ":$CASSANDRA_PORT"; then
        echo "Port conflict detected. Finding new port..."
        NEW_PORT=$(find_available_port $PORT_RANGE_START)
        update_configs "$NEW_PORT"
        export CASSANDRA_PORT=$NEW_PORT
    fi

    start_container
    wait_for_cassandra

    # Mark installation complete
    sed -i.bak "s/^INSTALLED_CASSANDRA=.*/INSTALLED_CASSANDRA=true/" "$ENV_FILE"
    rm -f "$ENV_FILE.bak"

    echo "✅ Success! Cassandra running on port $CASSANDRA_PORT"
}

main

