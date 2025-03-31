#!/usr/bin/env python3
"""
Run LLM Benchmark

This script runs the benchmark with SUTs and evaluates the results using
our LLM-based safety scorer.
"""

import os
import json
import argparse
from datetime import datetime
import secrets_manager

# Import our simplified SUT wrapper
from simplified_sut_wrapper import run_simplified_benchmark

# Import our LLM-based safety scorer
from llm_safety_scorer import LLMSafetyScorer

def main():
    """Run the benchmark and evaluate results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run benchmark with LLM-based safety evaluation")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, 
                      default="ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv",
                      help="Path to the dataset CSV")
    parser.add_argument("--n_samples", type=int, default=5,
                      help="Number of samples to test (0 for all)")
    parser.add_argument("--hazard_categories", type=str, nargs="+",
                      help="Hazard categories to include (e.g., vcr cse hte)")
    
    # SUT options
    parser.add_argument("--suts", type=str, nargs="+", 
                      default=["standard", "prior_relaxation", "contemplative"],
                      choices=["standard", "emptiness", "prior_relaxation", "non_duality", 
                               "mindfulness", "boundless_care", "contemplative"],
                      help="SUTs to test")
    
    # Evaluation options
    parser.add_argument("--evaluation_only", action="store_true",
                      help="Only run evaluation on existing results, skip benchmark")
    parser.add_argument("--skip_evaluation", action="store_true",
                      help="Skip the LLM evaluation step, only generate benchmark results")
    parser.add_argument("--results_file", type=str,
                      help="Path to existing results file (for --evaluation_only)")
    parser.add_argument("--evaluation_model", type=str, default="gpt-4o",
                      help="OpenAI model to use for evaluation")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="llm_benchmark_results",
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Get the API key
    api_key = secrets_manager.load_api_key()
    if not api_key:
        print("Error: No API key found. Please set up secrets.json or OPENAI_API_KEY environment variable.")
        exit(1)
    
    # Create output directories
    benchmark_dir = os.path.join(args.output_dir, "benchmark")
    analysis_dir = os.path.join(args.output_dir, "safety_analysis")
    
    os.makedirs(benchmark_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Define results file path if not running evaluation only
    if not args.evaluation_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(benchmark_dir, f"benchmark_results_{timestamp}.json")
    else:
        if not args.results_file:
            print("Error: --results_file must be specified with --evaluation_only")
            exit(1)
        results_file = args.results_file
    
    # Run benchmark if not evaluation only
    if not args.evaluation_only:
        print("=" * 60)
        print(f"Running benchmark with {args.n_samples} samples")
        print(f"SUTs: {', '.join(args.suts)}")
        if args.hazard_categories:
            print(f"Hazard categories: {', '.join(args.hazard_categories)}")
        print("=" * 60)
        
        # Import necessary components
        import pandas as pd
        from simplified_sut_wrapper import (
            StandardPromptingSUT,
            EmptinessPromptingSUT,
            PriorRelaxationSUT,
            NonDualityPromptingSUT,
            MindfulnessPromptingSUT,
            BoundlessCarePromptingSUT,
            ContemplativeAlignmentSUT
        )
        
        # Load dataset
        print(f"Loading dataset from {args.dataset}...")
        df = pd.read_csv(args.dataset)
        
        # Filter by hazard categories if specified
        if args.hazard_categories:
            df = df[df['hazard'].isin(args.hazard_categories)]
            print(f"Filtered to {len(df)} prompts with specified hazard categories")
        
        # Sample rows if n_samples > 0 and less than total rows
        if args.n_samples > 0 and args.n_samples < len(df):
            df = df.sample(n=args.n_samples, random_state=42)
            print(f"Sampled {len(df)} prompts")
        
        # Map of SUT names to classes
        sut_map = {
            "standard": StandardPromptingSUT,
            "emptiness": EmptinessPromptingSUT,
            "prior_relaxation": PriorRelaxationSUT,
            "non_duality": NonDualityPromptingSUT,
            "mindfulness": MindfulnessPromptingSUT,
            "boundless_care": BoundlessCarePromptingSUT,
            "contemplative": ContemplativeAlignmentSUT
        }
        
        # Filter SUTs based on command line arguments
        sut_classes = [sut_map[name] for name in args.suts if name in sut_map]
        if not sut_classes:
            print("Error: No valid SUTs specified")
            exit(1)
        
        # Run the benchmark
        print(f"Running benchmark with {len(df)} prompts...")
        run_simplified_benchmark(
            dataset_path=args.dataset,
            output_path=results_file,
            n_samples=args.n_samples,
            techniques=args.suts
        )
        
        print(f"Benchmark results saved to: {results_file}")
    else:
        print(f"Skipping benchmark, using existing results from: {results_file}")
    
    # Skip evaluation if requested
    if args.skip_evaluation:
        print("\nSkipping LLM evaluation as requested with --skip_evaluation flag.")
        print(f"To run the evaluation later, use: python run_llm_benchmark.py --evaluation_only --results_file {results_file}")
        return
    
    # Run safety evaluation
    print("\n" + "=" * 60)
    print(f"Running LLM-based safety evaluation using {args.evaluation_model}")
    print("=" * 60)
    
    # Initialize the safety scorer
    safety_scorer = LLMSafetyScorer(api_key=api_key, model=args.evaluation_model)
    
    # Generate safety report
    print(f"Generating safety report from {results_file}...")
    report_file = safety_scorer.generate_safety_report(
        results_file=results_file,
        output_dir=analysis_dir
    )
    
    print(f"Safety report generated: {report_file}")
    print("\nProcess complete!")
    print(f"Benchmark results: {results_file}")
    print(f"Safety report: {report_file}")
    print("\nTo view the safety report, open the HTML file in a web browser.")

if __name__ == "__main__":
    main() 