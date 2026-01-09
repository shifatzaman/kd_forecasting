#!/bin/bash

# Setup and run script for the improved forecasting model

echo "=========================================="
echo "KD Forecasting - Setup and Run"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the experiment
echo ""
echo "=========================================="
echo "Running experiments on both datasets"
echo "=========================================="
python3 run_both_datasets.py

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
