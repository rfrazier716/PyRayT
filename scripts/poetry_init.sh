#!/bin/bash

# Set some Env variables for the installation
export POETRY_HOME="/root/project/.poetry"
export POETRY="$POETRY_HOME/bin/poetry"

# Export these settings so they're available to other scripts
touch $BASH_ENV # touch the bash_env file to verify it exists
echo "export POETRY_HOME=$POETRY_HOME" >> $BASH_ENV
echo "export POETRY=$POETRY" >> $BASH_ENV

# Install Poetry and Verify Installation
echo "Installing Poetry"
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
if $POETRY --version ; then 
    $POETRY config virtualenvs.in-project true
    echo "Installed"
    exit 0 
else
    echo "Could not Install Poetry"
    exit 1
fi
