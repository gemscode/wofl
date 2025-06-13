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

K3S_CONTAINER="k3s-server"
K3S_NETWORK="k3s-net"
K3S_VOLUME="k3s-data"
K3S_IMAGE="rancher/k3s:v1.29.5-k3s1"
ENV_FILE="$(pwd)/.env"
KUBECONFIG_PATH="$(pwd)/k3s.yaml"
MAX_WAIT=300
DEFAULT_PORT=6443

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
    export K3S_HOST=${K3S_HOST:-$(detect_ip)}
    export K3S_PORT=${K3S_PORT:-$DEFAULT_PORT}
    export KUBECONFIG=${KUBECONFIG:-$KUBECONFIG_PATH}

    # Update .env with current values
    grep -q "^K3S_HOST=" "$ENV_FILE" || echo "K3S_HOST=$K3S_HOST" >> "$ENV_FILE"
    grep -q "^K3S_PORT=" "$ENV_FILE" || echo "K3S_PORT=$K3S_PORT" >> "$ENV_FILE"
    grep -q "^KUBECONFIG=" "$ENV_FILE" || echo "KUBECONFIG=$KUBECONFIG" >> "$ENV_FILE"
}

detect_ip() {
    local ip
    if command -v ip >/dev/null; then
        ip=$(ip route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if ($i=="src") print $(i+1)}')
    else
        ip=$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -n1)
    fi
    echo "${ip:-127.0.0.1}"
}

# --------------------------------------------------
# Port Management
# --------------------------------------------------
find_available_port() {
    local base_port=$1
    while true; do
        if ! nc -z "$K3S_HOST" "$base_port"; then
            echo "$base_port"
            return
        fi
        ((base_port++))
    done
}

update_port_config() {
    local new_port=$1
    sed -i.bak "s/^K3S_PORT=.*/K3S_PORT=$new_port/" "$ENV_FILE"
    rm -f "${ENV_FILE}.bak"
}

# --------------------------------------------------
# Docker Resource Management
# --------------------------------------------------
create_network() {
    if ! docker network inspect "$K3S_NETWORK" &>/dev/null; then
        echo "🐳 Creating Docker network: $K3S_NETWORK"
        docker network create "$K3S_NETWORK"
    fi
}

create_volume() {
    if ! docker volume inspect "$K3S_VOLUME" &>/dev/null; then
        echo "💾 Creating Docker volume: $K3S_VOLUME"
        docker volume create "$K3S_VOLUME"
    fi
}

# --------------------------------------------------
# Cluster Management
# --------------------------------------------------
manage_cluster() {
    # Handle port conflicts
    if nc -z "$K3S_HOST" "$K3S_PORT"; then
        echo "🚨 Port $K3S_PORT in use, finding alternative..."
        NEW_PORT=$(find_available_port $K3S_PORT)
        update_port_config "$NEW_PORT"
        export K3S_PORT=$NEW_PORT
    fi

    # Clean stale configuration
    if [ -f "$KUBECONFIG_PATH" ]; then
        local current_ip=$(grep 'server:' "$KUBECONFIG_PATH" | awk -F'//' '{print $2}' | cut -d: -f1)
        if [ "$current_ip" != "$K3S_HOST" ]; then
            echo "🔄 IP changed from $current_ip to $K3S_HOST - resetting cluster..."
            docker rm -f "$K3S_CONTAINER" 2>/dev/null || true
            docker volume rm "$K3S_VOLUME" 2>/dev/null || true
            rm -f "$KUBECONFIG_PATH"
        fi
    fi

    # Manage container state
    if docker ps -a --format '{{.Names}}' | grep -q "^${K3S_CONTAINER}$"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$K3S_CONTAINER")" = "false" ]; then
            echo "🚀 Starting existing k3s server..."
            docker start "$K3S_CONTAINER"
        else
            echo "✅ k3s server already running"
        fi
    else
        echo "🌟 Creating new k3s server..."
        docker run -d \
            --name "$K3S_CONTAINER" \
            --privileged \
            -p "${K3S_PORT}:${K3S_PORT}" \
            -v "${K3S_VOLUME}:/var/lib/rancher/k3s" \
            --network "$K3S_NETWORK" \
            "$K3S_IMAGE" server \
            --node-name "$K3S_CONTAINER" \
            --tls-san "$K3S_HOST" \
            --tls-san 127.0.0.1 \
            --disable=traefik \
            --disable=metrics-server \
            --https-listen-port "$K3S_PORT"
    fi
}

# --------------------------------------------------
# Health Checks
# --------------------------------------------------
wait_for_cluster() {
    echo "⏳ Waiting for k3s cluster readiness..."
    local counter=0
    until docker exec "$K3S_CONTAINER" kubectl get nodes &>/dev/null; do
        sleep 5
        counter=$((counter + 5))
        if [ $counter -ge $MAX_WAIT ]; then
            echo "❌ k3s cluster initialization timed out"
            exit 1
        fi
    done
}

# --------------------------------------------------
# Kubeconfig Management
# --------------------------------------------------
generate_kubeconfig() {
    echo "🔧 Generating kubeconfig..."
    docker cp "${K3S_CONTAINER}:/etc/rancher/k3s/k3s.yaml" "$KUBECONFIG_PATH"

    # Update server address using cross-platform sed
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "s|server:.*|server: https://${K3S_HOST}:${K3S_PORT}|g" "$KUBECONFIG_PATH"
    else
        sed -i "" "s|server:.*|server: https://${K3S_HOST}:${K3S_PORT}|g" "$KUBECONFIG_PATH"
    fi

    # Set context
    kubectl config --kubeconfig="$KUBECONFIG_PATH" rename-context default k3s-context || true
    echo "export KUBECONFIG=${KUBECONFIG_PATH}" >> "$ENV_FILE"
}

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
main() {
    echo "🚀 Starting k3s cluster initialization..."
    load_env
    create_network
    create_volume
    manage_cluster
    wait_for_cluster
    generate_kubeconfig

    echo ""
    echo "🎉 Kubernetes cluster initialized successfully!"
    echo "   API Server: https://${K3S_HOST}:${K3S_PORT}"
    echo "   Kubeconfig: ${KUBECONFIG_PATH}"
    echo "   Try: kubectl --kubeconfig=${KUBECONFIG_PATH} get nodes"
}

main "$@"

