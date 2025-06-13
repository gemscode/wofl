#!/bin/bash
# install_libev.sh (final version)

# Get script directory (works in packaged environment)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Paths relative to packaged location
LIBEV_SRC="$SCRIPT_DIR/lib/libev-4.33"
INSTALL_DIR="$SCRIPT_DIR/lib/libev-install"

# Clean and compile
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

cd "$LIBEV_SRC" || { echo "❌ libev source missing at $LIBEV_SRC"; exit 1; }
./configure --prefix="$INSTALL_DIR"
make && make install

# Install Cassandra driver with bundled libev
CFLAGS="-I$INSTALL_DIR/include" LDFLAGS="-L$INSTALL_DIR/lib" \
    pip install --no-binary :all: cassandra-driver

