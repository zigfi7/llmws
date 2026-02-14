#!/bin/bash
# LLMWS Status Script

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              LLMWS - Server Status                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo -e ""

# Check if venv exists
if [ -d "venv" ]; then
    echo -e "${GREEN}[✓] Virtual environment: OK${NC}"
else
    echo -e "${RED}[✗] Virtual environment: NOT FOUND${NC}"
    echo -e "${YELLOW}    Run ./start.sh to set up${NC}"
fi

# Check if server is running
PIDS=$(pgrep -f "python.*llmws.py")
if [ ! -z "$PIDS" ]; then
    echo -e "${GREEN}[✓] Server status: RUNNING${NC}"
    echo -e "${CYAN}    PIDs: ${PIDS}${NC}"
    
    # Show port
    PORT=$(netstat -tlnp 2>/dev/null | grep "$PIDS" | grep -oP ':\d+' | head -1 | cut -d':' -f2)
    if [ ! -z "$PORT" ]; then
        echo -e "${CYAN}    Port: ${PORT}${NC}"
    fi
    
    # Show uptime
    for PID in $PIDS; do
        UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
        echo -e "${CYAN}    Uptime: ${UPTIME}${NC}"
    done
else
    echo -e "${YELLOW}[!] Server status: STOPPED${NC}"
fi

# Check models directory
if [ -d "models" ]; then
    MODEL_COUNT=$(find models -maxdepth 1 -type d | tail -n +2 | wc -l)
    echo -e "${GREEN}[✓] Models directory: OK${NC}"
    echo -e "${CYAN}    Models found: ${MODEL_COUNT}${NC}"
    
    if [ $MODEL_COUNT -gt 0 ]; then
        echo -e "${CYAN}    Available models:${NC}"
        find models -maxdepth 1 -type d | tail -n +2 | while read dir; do
            name=$(basename "$dir")
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo -e "${CYAN}      - ${name} (${size})${NC}"
        done
    fi
else
    echo -e "${YELLOW}[!] Models directory: NOT FOUND${NC}"
fi

# Check var directory
if [ -d "var/sessions" ]; then
    SESSION_COUNT=$(find var/sessions -maxdepth 1 -type d | tail -n +2 | wc -l)
    if [ $SESSION_COUNT -gt 0 ]; then
        echo -e "${CYAN}[*] Active sessions: ${SESSION_COUNT}${NC}"
    fi
fi

# Check GPU
echo -e ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Status:${NC}"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read name mem_used mem_total gpu_util; do
        echo -e "${CYAN}  Device: ${name}${NC}"
        echo -e "${CYAN}  Memory: ${mem_used} MB / ${mem_total} MB${NC}"
        echo -e "${CYAN}  GPU Utilization: ${gpu_util}%${NC}"
    done
else
    echo -e "${YELLOW}GPU: Not available (CPU mode)${NC}"
fi
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e ""
