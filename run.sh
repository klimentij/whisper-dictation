#!/bin/bash
echo "$(dirname "$0")"
source .venv/bin/activate
python whisper-dictation.py --model_name turbo --k_double_cmd
