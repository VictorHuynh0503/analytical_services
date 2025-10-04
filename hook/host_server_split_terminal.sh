#!/bin/bash
echo "======================================="
echo " Starting DuckDB API services in tmux... "
echo "======================================="

# Kill any old session
tmux kill-session -t duckdb_services 2>/dev/null

# Create a new tmux session in detached mode
tmux new-session -d -s duckdb_services

# Window 1: sport_188bet (port 8000)
tmux send-keys -t duckdb_services "cd /root/sport_agents/app/services/sport_188bet && /root/selenium-env/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000" C-m

# Split pane and run second service
tmux split-window -h -t duckdb_services
tmux send-keys -t duckdb_services "cd /root/analytical_services/src/utils && /root/selenium-env/bin/python -m uvicorn telegram_webhook:app --host 0.0.0.0 --port 8001" C-m 

# Attach to tmux so you can see both
tmux attach -t duckdb_services

