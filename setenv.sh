#!/bin/bash
"""
WolfXE Environment Setup Script
Creates virtual environment and installs dependencies
"""

set -e  # Exit on any error

VENV_NAME="wolfxe_env"
REQUIREMENTS_FILE="requirements.txt"

echo "WolfXE Environment Setup"
echo "========================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: $REQUIREMENTS_FILE not found. Please ensure it exists in the current directory."
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

# Create new virtual environment
echo "Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To run the WolfXE simulator:"
echo "  python main.py --symbol AAPL --budget 25000 --speed 5.0"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
