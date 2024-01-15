#!/bin/bash

PORT_INFILLING=12345
PORT_GENERATION=12346

DEVICE_INFILLING=cuda:0
DEVICE_GENERATION=cuda:1

MODEL_INFILLING=codellama/CodeLlama-7b-Instruct-hf

MODEL_GENERATION=WizardLM/WizardCoder-Python-13B-V1.0
# uncomment the following if you want to test 34B version
# MODEL_GENERATION=WizardLM/WizardCoder-Python-34B-V1.0


case "$1" in
    --build-generation-server)
        python jumpcoder/run_llm_server.py --checkpoint $MODEL_GENERATION --port $PORT_GENERATION --device $DEVICE_GENERATION
        ;;
    
    --build-infilling-server)
        python jumpcoder/run_llm_server.py --checkpoint $MODEL_INFILLING --port $PORT_INFILLING --device $DEVICE_INFILLING --bad_words "#"
        ;;
    
    --evaluate)
        python evaluate.py --port_infilling $PORT_INFILLING --port_generation $PORT_GENERATION --dataset_path dataset/wizardcoder_mbpp.json --prompt_format wizardcoder --save_name results/mbpp-wc-13b.json --speculative_infill --parallel_generate_with_infill --threshold_improvement 0.3 --n_max_lines 128 --infill_comment
        ;;
esac