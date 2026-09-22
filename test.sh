#!/usr/bin/env bash
# ==============================================================================
# BCSE306L DA-1: Verification Test Suite
# Tests: 5-Module Closed-Loop Pipeline + Baselines/LOSO + Neural Retrainer
# ==============================================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}=== BCSE306L DA-1: Multimodal Cobot Trust System Verification ===${NC}\n"

echo -e "${CYAN}[1/3]${NC} Testing 5-Module Multimodal Closed-Loop Pipeline..."
python3 -c "import trust_ai_pipeline; trust_ai_pipeline.demo_system_run()"
echo -e "${GREEN}✔ 5-Module Pipeline Inference Passed.${NC}\n"

echo -e "${CYAN}[2/3]${NC} Evaluating Baselines, Ablations & LOSO Cross-Validation..."
python3 evaluation.py
echo -e "${GREEN}✔ Academic Baselines & Cross-Validation Passed.${NC}\n"

echo -e "${CYAN}[3/3]${NC} Running PyTorch AdamW Neural Calibration & Retraining..."
python3 retrainer.py
echo -e "${GREEN}✔ PyTorch AdamW Neural Retraining Passed.${NC}\n"

echo -e "================================================================"
echo -e "${GREEN}${BOLD}✔ ALL VERIFICATION TESTS PASSED SUCCESSFULLY!${NC}"
echo -e "System is ready for evaluation and live demonstration."
echo -e "================================================================\n"
