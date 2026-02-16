#!/bin/bash
# LLMWS Stop Script

set -uo pipefail

# Colors
if [ -t 1 ]; then
    G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; N='\033[0m'
else
    G=''; C=''; Y=''; N=''
fi

echo -e "${C}Stopping LLMWS server...${N}"

PIDS=$(pgrep -f "python.*llmws.py" || true)

if [ -z "$PIDS" ]; then
    echo -e "${Y}  No LLMWS server processes found${N}"
    exit 0
fi

echo -e "${C}  Found PIDs: ${PIDS}${N}"

for PID in $PIDS; do
    kill -TERM "$PID" 2>/dev/null || true
done

sleep 2

STILL_RUNNING=$(pgrep -f "python.*llmws.py" || true)
if [ -n "$STILL_RUNNING" ]; then
    echo -e "${Y}  Forcing remaining processes...${N}"
    for PID in $STILL_RUNNING; do
        kill -9 "$PID" 2>/dev/null || true
    done
fi

echo -e "${G}  LLMWS server stopped${N}"
