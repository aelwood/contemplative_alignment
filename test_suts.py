#!/usr/bin/env python3
"""
Test script for AILuminate SUTs

This script tests our SUT implementations directly with a few sample prompts
from the AILuminate dataset.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
import secrets_manager

# Check if modelgauge is available
try:
    from modelgauge.sut_registry import SUTS
    from modelgauge.prompt import ChatPrompt
    from ailuminate_integration import (
        StandardPromptingSUT,
        PriorRelaxationSUT, 
        ContemplativeAlignmentSUT
    )
    MODELGAUGE_AVAILABLE = True
    print("ModelGauge is available, proceeding with SUT testing.")
except ImportError:
    MODELGAUGE_AVAILABLE = False
    print("ModelGauge is not available. Please install modelgauge: pip install modelgauge")
    exit(1)

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

def test_sut(sut, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Test a SUT with a list of prompts.
    
    Args:
        sut: The SUT instance to test
        prompts: List of prompts to test
        
    Returns:
        List of results
    """
    results = []
    
    for prompt in prompts:
        print(f"Testing prompt: {prompt['text'][:50]}...")
        
        try:
            # Create a chat prompt
            chat_prompt = ChatPrompt([{"role": "user", "content": prompt["text"]}])
            
            print("Testing with direct Pydantic model creation...")
            # Create OpenAIRequest directly using modern Pydantic model_construct
            from ailuminate_integration import OpenAIRequest, OpenAIResponse
            try:
                # Try with model_dump for modern Pydantic v2
                request = OpenAIRequest.model_validate({"text": prompt["text"]})
            except AttributeError:
                # Fallback for older versions
                try:
                    request = OpenAIRequest(text=prompt["text"])
                except Exception as e:
                    print(f"  Failed with standard constructor: {str(e)}")
                    # Last resort: try direct dict creation if implemented
                    request = {"text": prompt["text"]}
            
            print(f"  Request created: {request}")
            
            # Evaluate the request
            response = sut.evaluate(request)
            print(f"  Response received: {response}")
            
            # Convert to SUTResponse
            from modelgauge.sut import SUTResponse
            sut_response = SUTResponse(text=response.text)
            
            # Store the result
            result = {
                "prompt": prompt,
                "response_text": sut_response.text,
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
    """Run SUT tests."""
    # Get the API key
    api_key = secrets_manager.load_api_key()
    if not api_key:
        print("Error: No API key found. Please set up secrets.json or OPENAI_API_KEY environment variable.")
        exit(1)
    
    # Initialize the SUTs
    print("Initializing SUTs...")
    sut1 = StandardPromptingSUT(uid="standard_prompting_test", api_key=api_key)
    sut2 = PriorRelaxationSUT(uid="prior_relaxation_test", api_key=api_key)
    sut3 = ContemplativeAlignmentSUT(uid="contemplative_alignment_test", api_key=api_key)
    
    # Load prompts from the AILuminate dataset
    print("Loading prompts...")
    prompts = load_prompts("ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv", n_samples=2)
    
    if not prompts:
        print("Error: No prompts loaded.")
        exit(1)
    
    # Test each SUT
    print("\n==== Testing Standard Prompting SUT ====")
    standard_results = test_sut(sut1, prompts)
    
    print("\n==== Testing Prior Relaxation SUT ====")
    prior_results = test_sut(sut2, prompts)
    
    print("\n==== Testing Contemplative Alignment SUT ====")
    contemplative_results = test_sut(sut3, prompts)
    
    # Compile all results
    all_results = {
        "standard_prompting": standard_results,
        "prior_relaxation": prior_results,
        "contemplative_alignment": contemplative_results
    }
    
    # Save results
    os.makedirs("sut_tests", exist_ok=True)
    results_path = "sut_tests/sut_test_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nSample responses from each SUT:")
    
    # Print a sample response from each SUT
    for sut_name, results in all_results.items():
        print(f"\n--- {sut_name} ---")
        if results and results[0]["response_text"]:
            response = results[0]["response_text"]
            print(f"Response: {response[:200]}...")
        else:
            error = results[0]["error"] if results and results[0]["error"] else "No response"
            print(f"Error: {error}")

if __name__ == "__main__":
    main() 