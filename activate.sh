#!/bin/bash
# Quick activation script for WolfXE environment

if [ -d "wolfxe_env" ]; then
    source wolfxe_env/bin/activate
    echo "WolfXE environment activated"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
else
    echo "ERROR: WolfXE environment not found. Run ./setenv.sh first."
fi
