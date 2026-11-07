#!/bin/bash
cd "$(dirname "$0")"
./.venv/bin/python run.py --steps 0 "$@"
