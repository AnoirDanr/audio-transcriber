#!/bin/bash

source $HOME/venvs/whisper/venv/bin/activate

python3 /usr/local/bin/whisper.py $1 $2

deactivate
