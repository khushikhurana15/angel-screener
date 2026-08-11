#!/bin/bash
while true; do
    echo "Starting dashboard..."
    python dashboard.py
    echo "Dashboard exited/crashed. Waiting 15 seconds before restart (avoids hammering rate limits)..."
    sleep 15
done