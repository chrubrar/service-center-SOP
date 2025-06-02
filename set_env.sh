#!/bin/bash
# Script to set environment variables for the SOP chatbot

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Activate the virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Added current directory to PYTHONPATH: $(pwd)"

# Run the specified command with the environment variables set and warning filter
if [ $# -eq 0 ]; then
    echo "Usage: source set_env.sh"
    echo "       # or to run a command:"
    echo "       ./set_env.sh <command>"
    echo ""
    echo "Examples:"
    echo "       ./set_env.sh python rebuild_vector_store.py"
    echo "       ./set_env.sh python test_layout_model.py localdata/SOPs/ADS\ process\ flows.pdf"
else
    # Create a temporary Python script that will be prepended to the target script
    TEMP_SCRIPT=$(mktemp)
    echo 'import warnings' > $TEMP_SCRIPT
    echo 'warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")' >> $TEMP_SCRIPT
    echo 'import sys' >> $TEMP_SCRIPT
    echo 'import os' >> $TEMP_SCRIPT
    echo 'sys.path.insert(0, os.getcwd())  # Add current directory to Python path' >> $TEMP_SCRIPT
    
    # Get the Python interpreter (first argument) and script (second argument)
    PYTHON_CMD=$1
    SCRIPT_PATH=$2
    shift 2  # Remove the first two arguments
    
    # Append the original script content to our temporary script
    cat $SCRIPT_PATH >> $TEMP_SCRIPT
    
    # Execute the combined script with the remaining arguments
    $PYTHON_CMD $TEMP_SCRIPT "$@"
    
    # Clean up
    rm $TEMP_SCRIPT
fi
