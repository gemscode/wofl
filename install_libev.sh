#!/bin/bash
# install_libev.sh

# Absolute paths (resolves symlinks for macOS)
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIB_DIR="${PROJECT_ROOT}/lib"
LIBEV_SRC="${PROJECT_ROOT}/lib/libev-4.33"
INSTALL_DIR="${PROJECT_ROOT}/lib/libev-install"

# Clean previous installations
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

# 1. Uninstall existing cassandra-driver first (globally and in venv) and download libenv
pip uninstall -y cassandra-driver &>/dev/null || true
pip cache remove cassandra_driver &>/dev/null || true
mkdir -p ${LIB_DIR}
wget -qO- https://dist.schmorp.de/libev/libev-4.33.tar.gz | tar xvz -C ${LIB_DIR}

# 2. Compile libev with absolute paths
cd "${LIBEV_SRC}" || { echo "❌ libev source missing at ${LIBEV_SRC}"; exit 1; }

./configure --prefix="${INSTALL_DIR}" --disable-dependency-tracking
make clean
make
make install

# 3. Set environment variables for current shell and future sessions
export DYLD_FALLBACK_LIBRARY_PATH="${INSTALL_DIR}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-/usr/lib}"
export C_INCLUDE_PATH="${INSTALL_DIR}/include:${C_INCLUDE_PATH}"

# For zsh (macOS default shell)
{
    echo "export DYLD_FALLBACK_LIBRARY_PATH=\"${INSTALL_DIR}/lib:\${DYLD_FALLBACK_LIBRARY_PATH}\""
    echo "export C_INCLUDE_PATH=\"${INSTALL_DIR}/include:\${C_INCLUDE_PATH}\""
} >> ~/.zshrc

# 4. Install cassandra-driver with explicit libev linkage
CFLAGS="-I${INSTALL_DIR}/include" \
LDFLAGS="-L${INSTALL_DIR}/lib" \
pip install --no-binary :all: --force-reinstall --isolated cassandra-driver

# 5. Verify installation
python -c "from cassandra.io.libevreactor import LibevConnection; assert LibevConnection is not None, 'LibevConnection missing!'"
echo "✅ libev + cassandra-driver reinstalled successfully"

