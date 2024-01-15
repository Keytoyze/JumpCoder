#!/bin/bash

PORT_INFILLING=12345
PORT_GENERATION=12346

DEVICE_INFILLING=cuda:0
DEVICE_GENERATION=cuda:1

MODEL_INFILLING=codellama/CodeLlama-7b-Instruct-hf

MODEL_GENERATION=codellama/CodeLlama-7b-Instruct-hf
# uncomment the following if you want to test other CL models
# MODEL_GENERATION=codellama/CodeLlama-7b-Python-hf
# MODEL_GENERATION=codellama/CodeLlama-13b-Instruct-hf
# MODEL_GENERATION=codellama/CodeLlama-13b-Python-hf


case "$1" in
    --build-generation-server)
        python jumpcoder/run_llm_server.py --checkpoint $MODEL_GENERATION --port $PORT_GENERATION --device $DEVICE_GENERATION
        ;;
    
    --build-infilling-server)
        python jumpcoder/run_llm_server.py --checkpoint $MODEL_INFILLING --port $PORT_INFILLING --device $DEVICE_INFILLING
        ;;
    
    --evaluate)
        python evaluate.py --port_infilling $PORT_INFILLING --port_generation $PORT_GENERATION --dataset_path dataset/humaneval.json --save_name results/he-cl-7b-instruct.json --speculative_infill --parallel_generate_with_infill --threshold_improvement 0.3
        ;;
esac