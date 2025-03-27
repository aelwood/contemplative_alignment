#!/usr/bin/env python3
"""
Minimal test script for AILuminate SUTs that bypasses modelgauge's ChatPrompt.

This script directly tests our SUT implementations with a few sample prompts
without using the modelgauge ChatPrompt class which appears to have Pydantic compatibility issues.
"""

import os
import json
from typing import List, Dict, Any
import secrets_manager
import pandas as pd

# Import our implementation classes directly
from contemplative_alignment_benchmark import (
    StandardPrompting,
    PriorRelaxationPrompting,
    ContemplativeAlignmentPrompting
)

def load_prompts(dataset_path: str, n_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Load prompts from the AILuminate dataset.
    
    Args:
        dataset_path: Path to the AILuminate dataset
        n_samples: Number of prompts to load
        
    Returns:
        List of prompts with metadata
    """
    try:
        # Load the CSV file
        df = pd.read_csv(dataset_path)
        
        # Take a sample of prompts
        df_sample = df.sample(n=n_samples, random_state=42)
        
        # Convert to list of dictionaries
        prompts = []
        for _, row in df_sample.iterrows():
            prompt_dict = {
                "text": row["prompt_text"],
                "hazard": row.get("hazard", None),
                "persona": row.get("persona", None),
                "prompt_id": row.get("release_prompt_id", None)
            }
            prompts.append(prompt_dict)
        
        return prompts
    except Exception as e:
        print(f"Error loading prompts: {str(e)}")
        return []

def test_technique(technique, prompts: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """
    Test a prompting technique with a list of prompts.
    
    Args:
        technique: The prompting technique to test
        prompts: List of prompts to test
        api_key: OpenAI API key
        
    Returns:
        List of results
    """
    results = []
    
    for prompt in prompts:
        print(f"Testing prompt: {prompt['text'][:50]}...")
        
        try:
            # Apply the technique directly
            response_text, _, _ = technique.apply(
                prompt=prompt["text"],
                model="gpt-4",
                api_key=api_key
            )
            
            # Store the result
            result = {
                "prompt": prompt,
                "response_text": response_text,
                "error": None
            }
            
            results.append(result)
        except Exception as e:
            print(f"Error testing prompt: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Store the error
            result = {
                "prompt": prompt,
                "response_text": None,
                "error": str(e)
            }
            
            results.append(result)
    
    return results

def main():
    """Run technique tests directly."""
    # Get the API key
    api_key = secrets_manager.load_api_key()
    if not api_key:
        print("Error: No API key found. Please set up secrets.json or OPENAI_API_KEY environment variable.")
        exit(1)
    
    # Initialize the prompting techniques directly
    print("Initializing prompting techniques...")
    standard = StandardPrompting()
    prior_relaxation = PriorRelaxationPrompting()
    contemplative = ContemplativeAlignmentPrompting()
    
    # Load prompts from the AILuminate dataset
    print("Loading prompts...")
    prompts = load_prompts("ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv", n_samples=2)
    
    if not prompts:
        print("Error: No prompts loaded.")
        exit(1)
    
    # Test each technique
    print("\n==== Testing Standard Prompting ====")
    standard_results = test_technique(standard, prompts, api_key)
    
    print("\n==== Testing Prior Relaxation Prompting ====")
    prior_results = test_technique(prior_relaxation, prompts, api_key)
    
    print("\n==== Testing Contemplative Alignment Prompting ====")
    contemplative_results = test_technique(contemplative, prompts, api_key)
    
    # Compile all results
    all_results = {
        "standard_prompting": standard_results,
        "prior_relaxation": prior_results,
        "contemplative_alignment": contemplative_results
    }
    
    # Save results
    os.makedirs("technique_tests", exist_ok=True)
    results_path = "technique_tests/technique_test_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nSample responses from each technique:")
    
    # Print a sample response from each technique
    for technique_name, results in all_results.items():
        print(f"\n--- {technique_name} ---")
        if results and results[0]["response_text"]:
            response = results[0]["response_text"]
            print(f"Response: {response[:200]}...")
        else:
            error = results[0]["error"] if results and results[0]["error"] else "No response"
            print(f"Error: {error}")

if __name__ == "__main__":
    main() 