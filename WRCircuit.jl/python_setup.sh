#! /bin/bash

# * Check conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install conda and try again."
    exit
fi

conda create -n WRCircuit
conda activate WRCircuit

# * Install based on the CondaPkg.toml
chmod u+x ./condapkg2yml.py # * Make the condapkg2yml.py script executable
TMP="$(mktemp).yml" # * Create a temporary file to store the .yml file
./condapkg2yml.py CondaPkg.toml > $TMP # * Convert the CondaPkg.toml to a .yml file for conda
conda env update -n WRCircuit --file $TMP # * Install the dependencies in the WRCircuit environment
