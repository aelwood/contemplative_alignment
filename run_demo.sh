#!/bin/bash

# Quick Start Demo for AILuminate Benchmark

# Default settings
INTEGRATED=false
FORCE_COMPATIBILITY=false
LLM_EVALUATION=false
EVALUATION_MODE="standard"
NUM_SAMPLES=5
TECHNIQUES="standard emptiness prior_relaxation non_duality mindfulness boundless_care contemplative"

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --integrated)
            INTEGRATED=true
            shift 1
            ;;
        --force-compatibility)
            FORCE_COMPATIBILITY=true
            shift 1
            ;;
        --llm-evaluation)
            LLM_EVALUATION=true
            EVALUATION_MODE="llm"
            shift 1
            ;;
        --evaluation-mode)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                EVALUATION_MODE="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --techniques)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                TECHNIQUES="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --n-samples)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                NUM_SAMPLES="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --help|-h)
            echo "Usage: ./run_demo.sh [options]"
            echo ""
            echo "Options:"
            echo "  --integrated                Run the integrated benchmark"
            echo "  --force-compatibility       Force compatibility mode"
            echo "  --llm-evaluation            Use LLM-based safety evaluation"
            echo "  --evaluation-mode MODE      Set evaluation mode: standard, ailuminate, or llm"
            echo "  --techniques \"TECH1 TECH2\"  Specify techniques to test (space-separated)"
            echo "  --n-samples N              Number of samples to test (default: 5)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Available techniques:"
            echo "  standard, emptiness, prior_relaxation, non_duality,"
            echo "  mindfulness, boundless_care, contemplative"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f "secrets.json" ]; then
        echo "Using API key from secrets.json"
    else
        echo "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or create a secrets.json file."
        exit 1
    fi
fi

# Create output directories
mkdir -p results
mkdir -p analysis
mkdir -p analysis/safety_analysis
mkdir -p integrated_results
mkdir -p llm_benchmark_results

# Set benchmark mode based on options
if [ "$EVALUATION_MODE" = "ailuminate" ]; then
    echo "Running AILuminate benchmark with integrated ModelGauge..."
    
    # Check if modelgauge is installed
    if ! pip show modelgauge > /dev/null 2>&1; then
        echo "Error: ModelGauge package not found. Please install it: pip install modelgauge"
        echo "Note: You must apply through MLCommons for access to the official ModelGauge package."
        exit 1
    fi
    
    echo "Using AILuminate dataset: ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv"
    echo "Number of samples: $NUM_SAMPLES"
    echo "Testing techniques: $TECHNIQUES"
    
    # Run the integrated benchmark
    python run_integrated_benchmark.py \
        --dataset ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv \
        --n_samples $NUM_SAMPLES \
        --output_dir integrated_results \
        --safety_report

elif [ "$EVALUATION_MODE" = "llm" ]; then
    echo "Running LLM-based safety evaluation benchmark..."
    echo "Using AILuminate dataset: ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv"
    echo "Number of samples: $NUM_SAMPLES"
    echo "Testing techniques: $TECHNIQUES"
    
    # Convert techniques to array
    IFS=' ' read -r -a TECHNIQUES_ARRAY <<< "$TECHNIQUES"
    
    # Run the LLM benchmark
    python run_llm_benchmark.py \
        --dataset ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv \
        --n_samples $NUM_SAMPLES \
        --suts ${TECHNIQUES_ARRAY[@]} \
        --output_dir llm_benchmark_results
else
    # Default to standard evaluation
    echo "Running standard safety evaluation benchmark..."
    echo "Using AILuminate dataset: ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv"
    echo "Number of samples: $NUM_SAMPLES"
    echo "Testing techniques: $TECHNIQUES"
    
    # Convert techniques to array
    IFS=' ' read -r -a TECHNIQUES_ARRAY <<< "$TECHNIQUES"
    
    # Run the benchmark with standard SUTs
    python test_suts_only.py \
        --dataset ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv \
        --n_samples $NUM_SAMPLES \
        --suts ${TECHNIQUES_ARRAY[@]}
fi

echo "Benchmark complete!"

echo ""
echo "For other evaluation modes:"
echo "./run_demo.sh --evaluation-mode standard    # Standard safety evaluation"
echo "./run_demo.sh --evaluation-mode ailuminate  # AILuminate SUT evaluation"
echo "./run_demo.sh --evaluation-mode llm         # LLM-based safety evaluation"
echo ""
echo "To specify techniques:"
echo "./run_demo.sh --techniques \"standard baseline contemplative\""
echo ""
echo "For more options, see: ./run_demo.sh --help" 