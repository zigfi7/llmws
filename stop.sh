#!/bin/bash
# LLMWS Stop Script

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}[*] Stopping LLMWS server...${NC}"

# Find and kill llmws.py processes
PIDS=$(pgrep -f "python.*llmws.py")

if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}[!] No LLMWS server processes found${NC}"
    exit 0
fi

echo -e "${CYAN}[*] Found processes: ${PIDS}${NC}"

# Kill processes gracefully
for PID in $PIDS; do
    echo -e "${CYAN}[*] Stopping PID: ${PID}${NC}"
    kill -TERM $PID 2>/dev/null
done

# Wait for processes to stop
sleep 2

# Check if any still running
STILL_RUNNING=$(pgrep -f "python.*llmws.py")
if [ ! -z "$STILL_RUNNING" ]; then
    echo -e "${YELLOW}[!] Some processes still running, forcing...${NC}"
    for PID in $STILL_RUNNING; do
        kill -9 $PID 2>/dev/null
    done
fi

echo -e "${GREEN}[✓] LLMWS server stopped${NC}"
