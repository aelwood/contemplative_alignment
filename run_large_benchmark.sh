#!/bin/bash

# This script runs a comprehensive benchmark with 100 samples across all techniques,
# followed by evaluations using both LLM-based and standard safety evaluators.
# Note: This will take several hours to complete due to the large number of samples.

# Set variables
NUM_SAMPLES=100
TECHNIQUES="standard emptiness prior_relaxation non_duality mindfulness boundless_care contemplative"
OUTPUT_DIR="large_benchmark_results"
DATE=$(date +"%Y%m%d_%H%M%S")

echo "================================================"
echo "🚀 Starting Comprehensive Benchmark (100 samples)"
echo "================================================"
echo "This will test all 7 techniques on $NUM_SAMPLES prompts:"
echo "- standard prompting"
echo "- emptiness prompting"
echo "- prior relaxation prompting"
echo "- non-duality prompting" 
echo "- mindfulness prompting"
echo "- boundless care prompting"
echo "- contemplative alignment prompting"
echo ""
echo "WARNING: This benchmark will take several hours to complete."
echo "It will generate 7 × $NUM_SAMPLES = 700 responses in total."
echo ""
echo "Press Ctrl+C now to cancel, or wait 5 seconds to continue..."
sleep 5

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

# Step 2: Run the benchmark to generate responses for all 100 prompts across all techniques
# Note: This will take the longest time (several hours)
echo ""
echo "Step 1/3: Running benchmark to generate responses..."
echo "============================================================"

python run_llm_benchmark.py \
    --n_samples $NUM_SAMPLES \
    --suts $TECHNIQUES \
    --output_dir "$OUTPUT_DIR"

BENCHMARK_RESULTS="$OUTPUT_DIR/benchmark/benchmark_results_"$(ls -t "$OUTPUT_DIR/benchmark" | grep json | head -n 1 | cut -d_ -f3-)
echo "Benchmark results saved to: $BENCHMARK_RESULTS"

# Step 3: Run the LLM-based safety evaluation on the benchmark results
echo ""
echo "Step 2/3: Running LLM-based safety evaluation..."
echo "============================================================"

python run_llm_benchmark.py \
    --evaluation_only \
    --results_file "$BENCHMARK_RESULTS" \
    --output_dir "$OUTPUT_DIR"

# Step 4: Run the standard safety evaluation on the benchmark results
echo ""
echo "Step 3/3: Running standard safety evaluation..."
echo "============================================================"

python analyze_ailuminate_results.py \
    --results_file "$BENCHMARK_RESULTS" \
    --output_dir "$OUTPUT_DIR/standard_safety_analysis"

echo ""
echo "✅ Comprehensive benchmark complete!"
echo "Results can be found in the following locations:"
echo "- Benchmark responses: $BENCHMARK_RESULTS"
echo "- LLM-based safety report: $OUTPUT_DIR/llm_safety_analysis/llm_safety_report_*.html"
echo "- Standard safety report: $OUTPUT_DIR/standard_safety_analysis/safety_report_*.html"
echo ""
echo "To compare the results, open the HTML reports in your web browser." 