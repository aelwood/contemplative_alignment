#!/usr/bin/env python3
"""
Run Integrated Benchmark

This script runs the integrated benchmark using AILuminate's SUT-based evaluation
with our custom safety scoring. It requires modelgauge to be installed.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from ailuminate_integration import AILuminateBenchmark

def load_api_key():
    """Load the OpenAI API key from environment or secrets.json file."""
    # Try environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
        
    # Try secrets.json file
    try:
        with open("secrets.json", "r") as f:
            secrets = json.load(f)
            return secrets.get("openai_api_key")
    except Exception as e:
        print(f"Error loading secrets.json: {e}")
        
    return None

def main():
    """Run the integrated benchmark."""
    parser = argparse.ArgumentParser(description="Run AILuminate benchmark with prompting techniques.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to test.")
    parser.add_argument("--hazard_categories", nargs="+", default=None, help="Specific hazard categories to filter for.")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key for the benchmark.")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save results.")
    parser.add_argument("--safety_report", action="store_true", help="Generate safety report.")
    
    args = parser.parse_args()
    
    # Check if ModelGauge is available
    if not MODELGAUGE_AVAILABLE:
        print("Error: ModelGauge is not available. Please install modelgauge using:")
        print("pip install modelgauge")
        exit(1)
    
    # Find or create API key
    api_key = args.api_key or load_api_key()
    if not api_key:
        print("Error: OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass --api_key.")
        exit(1)
    
    # Create output directory for benchmark results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate a timestamp for the benchmark results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"ailuminate_results_{timestamp}.json")
    
    # Run the benchmark
    print(f"Running AILuminate benchmark on dataset: {args.dataset}")
    print(f"Number of samples: {args.n_samples}")
    if args.hazard_categories:
        print(f"Filtering for hazard categories: {args.hazard_categories}")
    
    print(f"API key found: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        # Initialize the AILuminate benchmark
        benchmark = AILuminateBenchmark(api_key=api_key)
        
        # Prepare the dataset
        df = pd.read_csv(args.dataset)
        
        # Filter by hazard categories if specified
        if args.hazard_categories:
            df = df[df["hazard"].isin(args.hazard_categories)]
        
        # Limit to n_samples
        if args.n_samples and args.n_samples < len(df):
            df = df.sample(n=args.n_samples, random_state=42)
        
        # Save the filtered dataset for the benchmark
        filtered_dataset = os.path.join(args.output_dir, f"filtered_dataset_{timestamp}.csv")
        df.to_csv(filtered_dataset, index=False)
        
        # Run the benchmark
        results = benchmark.run_ailuminate_benchmark(
            filtered_dataset, 
            output_file,
            num_samples=args.n_samples
        )
        
        print(f"\nBenchmark completed. Results saved to {output_file}")
        
        # If safety report is requested, generate it
        if args.safety_report:
            from safety_evaluator import SafetyEvaluator
            
            # Initialize the safety evaluator
            safety_evaluator = SafetyEvaluator()
            
            # Generate safety report from the results
            print("Generating safety report...")
            
            # Make sure results have the correct format for the safety evaluator
            if "samples" not in results and "results" in results:
                # Convert from old format to new format if needed
                print("Converting results to new format with samples...")
                new_results = {
                    "metadata": results.get("metadata", {}),
                    "samples": []
                }
                
                # Extract raw results and create samples format
                for sut_name, sut_results in results.get("results", {}).items():
                    if "test_results" in sut_results:
                        technique_name = sut_name.replace("_prompting", "").replace("_", "")
                        
                        for test_result in sut_results["test_results"]:
                            # Find or create a sample for this prompt
                            prompt_text = test_result.get("prompt", "")
                            existing_sample = None
                            
                            for sample in new_results["samples"]:
                                if sample["prompt_text"] == prompt_text:
                                    existing_sample = sample
                                    break
                            
                            if not existing_sample:
                                existing_sample = {
                                    "prompt_id": "unknown",
                                    "prompt_text": prompt_text,
                                    "hazard": test_result.get("hazard_category", None),
                                    "techniques": {}
                                }
                                new_results["samples"].append(existing_sample)
                            
                            # Add the technique response to the sample
                            existing_sample["techniques"][technique_name] = {
                                "response": test_result.get("response", ""),
                                "tokens_used": 0,
                                "time_taken": 0,
                                "error": None
                            }
                
                results = new_results
                
                # Save the converted results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            # Generate the safety report
            report_file = os.path.join(args.output_dir, f"safety_report_{timestamp}.html")
            safety_evaluator.generate_safety_report(results, report_file)
            
            print(f"Safety report generated at {report_file}")
    
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 