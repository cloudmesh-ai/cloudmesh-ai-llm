#!/usr/bin/env bash
set -e

# Visual colors for scannability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

HOSTS=("white" "spark")

clear
echo "=========================================================================="
echo "                     CLUSTER NETWORK & FIREWALL AUDIT                     "
echo "=========================================================================="

for HOST in "${HOSTS[@]}"; do
    # Convert hostname to uppercase using a universally compatible method
    HOST_UPPER=$(echo "$HOST" | tr '[:lower:]' '[:upper:]')

    echo -e "\n${BLUE}>>> CONNECTING TO NODE: ${HOST_UPPER} ...${NC}"
    echo "--------------------------------------------------------------------------"

    # Using -T drops the pseudo-terminal requirement, avoiding stdin warnings
    # Single quotes around 'EOF' tell bash to send the text exactly as written
    ssh -T "$HOST" "bash -s" << 'EOF'
# 1. Fetch IP allocations
INTERNAL_IP=$(hostname -I | awk '{print $1}')
EXTERNAL_IP=$(curl -s --max-time 4 ifconfig.me || echo "TIMEOUT")

echo -e "[+] Network Boundary Identity:"
echo -e "    - Primary Internal IP : ${YELLOW}${INTERNAL_IP}${NC}"
echo -e "    - Outbound Public IP  : ${YELLOW}${EXTERNAL_IP}${NC}"

# 2. Risk Evaluation
echo -e "\n[+] Perimeter Evaluation:"
if [ "${EXTERNAL_IP}" = "TIMEOUT" ]; then
    echo -e "    - ${YELLOW}Status:${NC} Could not fetch public IP. Host may be air-gapped or proxy-restricted."
elif [ "${INTERNAL_IP}" = "${EXTERNAL_IP}" ]; then
    echo -e "    - ${RED}CRITICAL EXPOSURE WARNING:${NC} Internal IP matches External IP."
    echo -e "      This host is directly attached to the public internet edge."
else
    echo -e "    - ${GREEN}SECURE FROM PUBLIC INTERNET:${NC} NAT translation detected (Internal != External)."
    echo -e "      Public web traffic targeting port 18000/18001 will drop at your gateway router."
fi

# 3. Local Subnet Check
echo -e "\n[+] Local Subnet Bound Interfaces:"
ip -4 -br addr show | grep -vE 'lo|br-|docker' | while read -r line; do
    echo "    - $line"
done

# 4. Active Firewall Rules Audit
echo -e "\n[+] Active Firewall Rules Audit:"
echo -e "    ${BLUE}[Sudo Check]${NC} Requesting administrative permissions to read host rule sets..."

if command -v ufw >/dev/null 2>&1; then
    UFW_STATUS=$(sudo ufw status 2>/dev/null || echo "REQUIRES_SUDO")
    
    if [ "${UFW_STATUS}" = "REQUIRES_SUDO" ]; then
        echo -e "    - ${YELLOW}UFW detected:${NC} Could not authenticate sudo privileges."
    elif echo "${UFW_STATUS}" | grep -q "Status: active"; then
        echo -e "    - ${GREEN}UFW Firewall:${NC} ACTIVE"
        VLLM_RULES=$(echo "${UFW_STATUS}" | grep -E '18000|18001|8000|8001' || true)
        if [ -n "${VLLM_RULES}" ]; then
            echo -e "      ${BLUE}Relevant Explicit Port Rules Found:${NC}"
            echo "${VLLM_RULES}" | sed 's/^/      /'
        else
            echo -e "      ${YELLOW}Note:${NC} No explicit rules found matching vLLM ports (8000, 8001, 18000, 18001)."
        fi
    else
        echo -e "    - ${RED}UFW Firewall:${NC} INACTIVE / DISABLED"
    fi
else
    echo -e "    - ${YELLOW}UFW:${NC} Not installed on this system configuration."
fi

# Check Raw iptables entries for explicit Docker network modifications
if command -v iptables >/dev/null 2>&1; then
    IPTABLES_RULES=$(sudo iptables -L INPUT -n -v 2>/dev/null | grep -E '18000|18001|8000|8001' || true)
    IPTABLES_DOCKER=$(sudo iptables -L DOCKER -n -v 2>/dev/null | grep -E '18000|18001|8000|8001' || true)
    
    if [ -n "${IPTABLES_RULES}" ]; then
        echo -e "      ${BLUE}Active INPUT Chain overrides:${NC}"
        echo "${IPTABLES_RULES}" | sed 's/^/      /'
    fi
    if [ -n "${IPTABLES_DOCKER}" ]; then
        echo -e "      ${RED}Active Docker Isolation Chain rules:${NC}"
        echo "${IPTABLES_DOCKER}" | sed 's/^/      /'
    fi
fi
EOF
    echo "--------------------------------------------------------------------------"
done

echo -e "\n${YELLOW}* Security Reminder:${NC} Even with NAT protection, binding Docker to 0.0.0.0 allows any"
echo "  other machine inside your local subnet to reach your vLLM endpoints."
echo "  To restrict access strictly to your local machine (SSH Tunnels only),"
echo "  ensure your Docker flags use: -p 127.0.0.1:18001:18001"
echo "=========================================================================="