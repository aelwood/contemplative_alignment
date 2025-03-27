#!/usr/bin/env python3
"""
Test SUTs Only

This script runs just the SUT testing without any safety analysis.
It provides command-line options for customizing which SUTs to test,
how many samples to use, and other parameters.
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import secrets_manager

# Import our prompting techniques
from contemplative_alignment_benchmark import (
    StandardPrompting,
    PriorRelaxationPrompting,
    ContemplativeAlignmentPrompting
)

# Import simplified SUT wrappers
from simplified_sut_wrapper import (
    SimplifiedSUT,
    StandardPromptingSUT,
    PriorRelaxationSUT,
    ContemplativeAlignmentSUT
)

def load_dataset(dataset_path: str, 
                n_samples: int = 5, 
                hazard_categories: Optional[List[str]] = None,
                random_seed: int = 42) -> pd.DataFrame:
    """
    Load and filter the dataset.
    
    Args:
        dataset_path: Path to the dataset
        n_samples: Number of samples to load (0 for all)
        hazard_categories: List of hazard categories to include
        random_seed: Random seed for sampling
        
    Returns:
        Filtered DataFrame
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Filter by hazard categories if specified
    if hazard_categories:
        df = df[df['hazard'].isin(hazard_categories)]
        
    # Sample rows if n_samples > 0 and less than total rows
    if n_samples > 0 and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed)
    
    return df

def test_suts(df: pd.DataFrame, 
              sut_names: List[str], 
              api_key: str,
              output_dir: str) -> str:
    """
    Test the specified SUTs on the dataset.
    
    Args:
        df: Dataset DataFrame
        sut_names: Names of SUTs to test
        api_key: OpenAI API key
        output_dir: Directory to save results
        
    Returns:
        Path to the results file
    """
    # Map of SUT names to classes
    sut_map = {
        "standard": StandardPromptingSUT,
        "prior_relaxation": PriorRelaxationSUT,
        "contemplative": ContemplativeAlignmentSUT
    }
    
    # Initialize the requested SUTs
    suts = []
    for name in sut_names:
        if name in sut_map:
            sut_class = sut_map[name]
            sut = sut_class(api_key=api_key)
            suts.append(sut)
        else:
            print(f"Warning: Unknown SUT '{name}', skipping")
    
    if not suts:
        raise ValueError("No valid SUTs specified")
    
    # Create the format for results
    samples = []
    technique_names = [sut.name for sut in suts]
    
    # Process each prompt with each SUT
    for _, row in df.iterrows():
        prompt_text = row["prompt_text"]
        prompt_id = row.get("release_prompt_id", None)
        hazard = row.get("hazard", None)
        
        print(f"Processing prompt: {prompt_text[:50]}...")
        
        # Create a sample entry
        sample = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "hazard": hazard,
            "techniques": {}
        }
        
        # Process with each SUT
        for sut in suts:
            print(f"  Testing with {sut.name}...")
            
            try:
                # Process the prompt
                response_text, metadata = sut.process_prompt(prompt_text)
                
                # Store the result
                sample["techniques"][sut.name] = {
                    "response": response_text,
                    "tokens_used": metadata["tokens_used"],
                    "time_taken": metadata["time_taken"],
                    "error": None
                }
                
                # Print a snippet of the response
                snippet = response_text[:100] + "..." if len(response_text) > 100 else response_text
                print(f"  Response: {snippet}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                
                # Store the error
                sample["techniques"][sut.name] = {
                    "response": f"ERROR: {str(e)}",
                    "tokens_used": 0,
                    "time_taken": 0,
                    "error": str(e)
                }
        
        # Add the sample to the list
        samples.append(sample)
    
    # Compile all results
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model": "gpt-4",
            "dataset": df.shape[0],
            "techniques_tested": technique_names
        },
        "samples": samples
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"sut_results_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return output_path

def main():
    """Run the SUTs-only test."""
    parser = argparse.ArgumentParser(description="Test SUTs without safety analysis")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, 
                      default="ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv",
                      help="Path to the dataset CSV")
    parser.add_argument("--n_samples", type=int, default=5,
                      help="Number of samples to test (0 for all)")
    parser.add_argument("--hazard_categories", type=str, nargs="+",
                      help="Hazard categories to include (e.g., vcr cse hte)")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for sampling")
    
    # SUT options
    parser.add_argument("--suts", type=str, nargs="+", 
                      default=["standard", "prior_relaxation", "contemplative"],
                      help="SUTs to test (standard, prior_relaxation, contemplative)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="sut_results",
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Get the API key
    api_key = secrets_manager.load_api_key()
    if not api_key:
        print("Error: No API key found. Please set up secrets.json or OPENAI_API_KEY environment variable.")
        exit(1)
    
    # Load and filter dataset
    print(f"Loading dataset from {args.dataset}...")
    df = load_dataset(
        dataset_path=args.dataset,
        n_samples=args.n_samples,
        hazard_categories=args.hazard_categories,
        random_seed=args.random_seed
    )
    
    if df.empty:
        print("Error: No prompts found after filtering")
        exit(1)
    
    print(f"Loaded {len(df)} prompts")
    
    # Test SUTs
    print(f"Testing SUTs: {', '.join(args.suts)}")
    results_path = test_suts(
        df=df,
        sut_names=args.suts,
        api_key=api_key,
        output_dir=args.output_dir
    )
    
    print(f"\nSUT testing complete!")
    print(f"Results saved to: {results_path}")
    print("\nTo view results:")
    print(f"  cat {results_path} | less")
    
if __name__ == "__main__":
    main() 