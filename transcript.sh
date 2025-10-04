#!/bin/bash

source $HOME/venvs/whisper/venv/bin/activate

python3 whisper.py $1

deactivate
