#!/usr/bin/env bash
# Sandy dashboard (streamlit multipage) on port 8502.
# Exposed as sandy.inprojectfitness.com via the existing Cloudflare tunnel.
# NOTE (1.9GB box): stop this before running heavy builds:  pkill -f "streamlit run"
cd /home/ec2-user/sandy
source "$HOME/.sandy_env"
exec .venv/bin/python -m streamlit run sandy/dashboard/Sandy.py \
    --server.port 8502 --server.address 127.0.0.1 --server.headless true \
    --browser.gatherUsageStats false
