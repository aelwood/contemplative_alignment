#!/bin/bash

# This script runs a comprehensive benchmark with 100 samples across all techniques,
# followed by evaluations using both LLM-based and standard safety evaluators.
# Note: This will take several hours to complete due to the large number of samples.

# Default settings
NUM_SAMPLES=100
TECHNIQUES="standard emptiness prior_relaxation non_duality mindfulness boundless_care contemplative"
OUTPUT_DIR="large_benchmark_results"
SKIP_BENCHMARK=false
SKIP_LLM_EVAL=false
SKIP_STANDARD_EVAL=false
DATE=$(date +"%Y%m%d_%H%M%S")

# Function to display usage information
function show_usage {
    echo "Usage: ./run_large_benchmark.sh [options]"
    echo ""
    echo "Options:"
    echo "  --samples N                   Number of samples to test (default: 100)"
    echo "  --techniques \"TECH1 TECH2...\" Techniques to test (default: all 7 techniques)"
    echo "  --output-dir DIR              Output directory (default: large_benchmark_results)"
    echo "  --skip-benchmark              Skip running the benchmark (use existing results)"
    echo "  --skip-llm-eval               Skip running the LLM-based evaluation"
    echo "  --skip-standard-eval          Skip running the standard safety evaluation"
    echo "  --help                        Display this help message"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --techniques)
            TECHNIQUES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift 1
            ;;
        --skip-llm-eval)
            SKIP_LLM_EVAL=true
            shift 1
            ;;
        --skip-standard-eval)
            SKIP_STANDARD_EVAL=true
            shift 1
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

echo "================================================"
echo "🚀 Running Comprehensive Benchmark and Evaluation"
echo "================================================"
echo "Settings:"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Techniques: $TECHNIQUES"
echo "  Output directory: $OUTPUT_DIR"
echo "  Skip benchmark: $SKIP_BENCHMARK"
echo "  Skip LLM evaluation: $SKIP_LLM_EVAL"
echo "  Skip standard evaluation: $SKIP_STANDARD_EVAL"
echo ""

if [ "$SKIP_BENCHMARK" = false ]; then
    echo "WARNING: The benchmark will take several hours to complete."
    echo "It will generate responses for $NUM_SAMPLES prompts × $(echo $TECHNIQUES | wc -w) techniques."
    echo ""
    echo "Press Ctrl+C now to cancel, or wait 5 seconds to continue..."
    sleep 5
fi

# Verify that secrets.json exists
if [ ! -f "secrets.json" ]; then
    echo "Error: secrets.json not found."
    echo "Please create a secrets.json file with your OpenAI API key."
    exit 1
fi

# Step 1: Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/benchmark"
mkdir -p "$OUTPUT_DIR/llm_safety_analysis"
mkdir -p "$OUTPUT_DIR/standard_safety_analysis"

# Look for existing benchmark results
EXISTING_RESULTS=$(ls -t "$OUTPUT_DIR/benchmark" 2>/dev/null | grep json | head -n 1)
if [ -n "$EXISTING_RESULTS" ]; then
    EXISTING_BENCHMARK="$OUTPUT_DIR/benchmark/$EXISTING_RESULTS"
    echo "Found existing benchmark results: $EXISTING_BENCHMARK"
fi

# Run the benchmark if not skipped
if [ "$SKIP_BENCHMARK" = false ]; then
    echo ""
    echo "Step 1/3: Running benchmark to generate responses..."
    echo "============================================================"

    # Note: We don't include --evaluation_only flag here to avoid immediately running the evaluation
    # The run_llm_benchmark.py script will still generate responses, but not perform the LLM evaluation yet
    python run_llm_benchmark.py \
        --n_samples $NUM_SAMPLES \
        --suts $TECHNIQUES \
        --output_dir "$OUTPUT_DIR" \
        --skip_evaluation

    # Find the most recent benchmark results file
    BENCHMARK_RESULTS="$OUTPUT_DIR/benchmark/"$(ls -t "$OUTPUT_DIR/benchmark" | grep json | head -n 1)
    echo "Benchmark results saved to: $BENCHMARK_RESULTS"
else
    if [ -n "$EXISTING_RESULTS" ]; then
        BENCHMARK_RESULTS="$EXISTING_BENCHMARK"
        echo "Using existing benchmark results: $BENCHMARK_RESULTS"
    else
        echo "Error: No existing benchmark results found and --skip-benchmark was specified."
        echo "Please run without --skip-benchmark first or specify an existing results file."
        exit 1
    fi
fi

# Check if LLM evaluation has already been performed
LLM_EVAL_EXISTS=false
if [ -d "$OUTPUT_DIR/llm_safety_analysis" ]; then
    LLM_EVAL_FILES=$(ls -t "$OUTPUT_DIR/llm_safety_analysis" 2>/dev/null | grep llm_safety_report | grep html | head -n 1)
    if [ -n "$LLM_EVAL_FILES" ]; then
        LLM_EVAL_EXISTS=true
        echo "Found existing LLM evaluation: $OUTPUT_DIR/llm_safety_analysis/$LLM_EVAL_FILES"
    fi
fi

# Run the LLM-based safety evaluation if not skipped
if [ "$SKIP_LLM_EVAL" = false ]; then
    if [ "$LLM_EVAL_EXISTS" = true ]; then
        echo ""
        echo "LLM evaluation already exists. Skipping..."
        echo "Use --skip-llm-eval to skip this step or manually delete existing reports to force reevaluation."
    else
        echo ""
        echo "Step 2/3: Running LLM-based safety evaluation..."
        echo "============================================================"

        python run_llm_benchmark.py \
            --evaluation_only \
            --results_file "$BENCHMARK_RESULTS" \
            --output_dir "$OUTPUT_DIR"
    fi
else
    echo ""
    echo "Skipping LLM-based safety evaluation as requested."
fi

# Check if standard evaluation has already been performed
STD_EVAL_EXISTS=false
if [ -d "$OUTPUT_DIR/standard_safety_analysis" ]; then
    STD_EVAL_FILES=$(ls -t "$OUTPUT_DIR/standard_safety_analysis" 2>/dev/null | grep safety_report | grep html | head -n 1)
    if [ -n "$STD_EVAL_FILES" ]; then
        STD_EVAL_EXISTS=true
        echo "Found existing standard evaluation: $OUTPUT_DIR/standard_safety_analysis/$STD_EVAL_FILES"
    fi
fi

# Run the standard safety evaluation if not skipped
if [ "$SKIP_STANDARD_EVAL" = false ]; then
    if [ "$STD_EVAL_EXISTS" = true ]; then
        echo ""
        echo "Standard evaluation already exists. Skipping..."
        echo "Use --skip-standard-eval to skip this step or manually delete existing reports to force reevaluation."
    else
        echo ""
        echo "Step 3/3: Running standard safety evaluation..."
        echo "============================================================"

        python analyze_ailuminate_results.py \
            --results_file "$BENCHMARK_RESULTS" \
            --output_dir "$OUTPUT_DIR/standard_safety_analysis"
    fi
else
    echo ""
    echo "Skipping standard safety evaluation as requested."
fi

echo ""
echo "✅ Process complete!"
echo "Results can be found in the following locations:"
echo "- Benchmark responses: $BENCHMARK_RESULTS"
echo "- LLM-based safety report: $OUTPUT_DIR/llm_safety_analysis/llm_safety_report_*.html"
echo "- Standard safety report: $OUTPUT_DIR/standard_safety_analysis/safety_report_*.html"
echo ""
echo "To compare the results, open the HTML reports in your web browser."
echo ""
echo "For future runs with these results, you can use:"
echo "./run_large_benchmark.sh --skip-benchmark --output-dir $OUTPUT_DIR" 