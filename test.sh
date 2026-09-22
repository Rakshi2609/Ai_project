#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Automated Pipeline & Model Verification Suite
# ==============================================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
TEAL='\033[38;5;43m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${TEAL}${BOLD}=== BCSE306L DA-1: Multimodal Cobot Trust System Verification ===${NC}\n"

echo -e "${CYAN}[1/3] Testing 5-Module Multimodal Closed-Loop Pipeline...${NC}"
python3 trust_ai_pipeline.py
echo -e "${GREEN}✔ 5-Module Pipeline Inference Passed.${NC}\n"

echo -e "${CYAN}[2/3] Evaluating Baselines, Ablations & LOSO Cross-Validation...${NC}"
python3 evaluation.py
echo -e "${GREEN}✔ Academic Baselines & Cross-Validation Passed.${NC}\n"

echo -e "${CYAN}[3/3] Running PyTorch AdamW Neural Calibration & Retraining...${NC}"
python3 retrainer.py
echo -e "${GREEN}✔ PyTorch AdamW Neural Retraining Passed.${NC}\n"

echo -e "${TEAL}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}✔ ALL VERIFICATION TESTS PASSED SUCCESSFULLY!${NC}"
echo -e "System is ready for evaluation and live demonstration."
echo -e "${TEAL}${BOLD}================================================================${NC}"
