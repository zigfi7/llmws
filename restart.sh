#!/bin/bash
# LLMWS Restart Script

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║            LLMWS - Restart Server                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo -e ""

# Stop server
./stop.sh

echo -e ""

# Wait a moment
sleep 1

# Start server
./start.sh
