#!/usr/bin/env python3
"""
Simplified SUT Wrapper for AILuminate Integration

This module provides a simplified wrapper for our prompting techniques that 
works around the Pydantic compatibility issues in the modelgauge package.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import secrets_manager

# Import our prompting techniques
from contemplative_alignment_benchmark import (
    StandardPrompting,
    EmptinessPrompting,
    PriorRelaxationPrompting,
    NonDualityPrompting,
    MindfulnessPrompting,
    BoundlessCarePrompting,
    ContemplativeAlignmentPrompting
)

class SimplifiedSUT:
    """Base class for simplified SUT wrappers that bypass modelgauge compatibility issues."""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        """
        Initialize the simplified SUT.
        
        Args:
            name: Name of the SUT
            api_key: OpenAI API key (optional if set as environment variable)
        """
        self.name = name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize technique - to be implemented by subclasses
        self.technique = None
    
    def process_prompt(self, prompt_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a prompt and return the response and metadata.
        
        Args:
            prompt_text: The prompt text to process
            
        Returns:
            Tuple of (response_text, metadata)
        """
        if not self.technique:
            raise ValueError("Technique not initialized")
        
        # Apply the technique
        response_text, tokens_used, time_taken = self.technique.apply(
            prompt=prompt_text,
            model="gpt-4",
            api_key=self.api_key
        )
        
        # Compile metadata
        metadata = {
            "tokens_used": tokens_used,
            "time_taken": time_taken,
            "sut_name": self.name
        }
        
        return response_text, metadata


class StandardPromptingSUT(SimplifiedSUT):
    """Simplified wrapper for Standard Prompting technique."""
    
    def __init__(self, name: str = "standard_prompting", api_key: Optional[str] = None):
        """Initialize with Standard Prompting technique."""
        super().__init__(name, api_key)
        self.technique = StandardPrompting()


class EmptinessPromptingSUT(SimplifiedSUT):
    """Simplified wrapper for Emptiness Prompting technique."""
    
    def __init__(self, name: str = "emptiness", api_key: Optional[str] = None):
        """Initialize with Emptiness Prompting technique."""
        super().__init__(name, api_key)
        self.technique = EmptinessPrompting()


class PriorRelaxationSUT(SimplifiedSUT):
    """Simplified wrapper for Prior Relaxation Prompting technique."""
    
    def __init__(self, name: str = "prior_relaxation", api_key: Optional[str] = None):
        """Initialize with Prior Relaxation Prompting technique."""
        super().__init__(name, api_key)
        self.technique = PriorRelaxationPrompting()


class NonDualityPromptingSUT(SimplifiedSUT):
    """Simplified wrapper for Non-Duality Prompting technique."""
    
    def __init__(self, name: str = "non_duality", api_key: Optional[str] = None):
        """Initialize with Non-Duality Prompting technique."""
        super().__init__(name, api_key)
        self.technique = NonDualityPrompting()


class MindfulnessPromptingSUT(SimplifiedSUT):
    """Simplified wrapper for Mindfulness Prompting technique."""
    
    def __init__(self, name: str = "mindfulness", api_key: Optional[str] = None):
        """Initialize with Mindfulness Prompting technique."""
        super().__init__(name, api_key)
        self.technique = MindfulnessPrompting()


class BoundlessCarePromptingSUT(SimplifiedSUT):
    """Simplified wrapper for Boundless Care Prompting technique."""
    
    def __init__(self, name: str = "boundless_care", api_key: Optional[str] = None):
        """Initialize with Boundless Care Prompting technique."""
        super().__init__(name, api_key)
        self.technique = BoundlessCarePrompting()


class ContemplativeAlignmentSUT(SimplifiedSUT):
    """Simplified wrapper for Contemplative Alignment Prompting technique."""
    
    def __init__(self, name: str = "contemplative_alignment", api_key: Optional[str] = None):
        """Initialize with Contemplative Alignment Prompting technique."""
        super().__init__(name, api_key)
        self.technique = ContemplativeAlignmentPrompting()


def run_simplified_benchmark(dataset_path: str, output_path: str, n_samples: int = 5, techniques: List[str] = None) -> Dict[str, Any]:
    """
    Run a simplified benchmark using our SUTs directly.
    
    Args:
        dataset_path: Path to the dataset
        output_path: Path to save the results
        n_samples: Number of samples to test
        techniques: List of technique names to test (default: all)
        
    Returns:
        Dictionary with benchmark results
    """
    import pandas as pd
    from datetime import datetime
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Take a sample of rows
    if n_samples > 0 and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
    
    # Get the API key
    api_key = secrets_manager.load_api_key()
    
    # Map technique names to SUT classes
    sut_map = {
        "standard": StandardPromptingSUT,
        "emptiness": EmptinessPromptingSUT,
        "prior_relaxation": PriorRelaxationSUT,
        "non_duality": NonDualityPromptingSUT,
        "mindfulness": MindfulnessPromptingSUT,
        "boundless_care": BoundlessCarePromptingSUT,
        "contemplative": ContemplativeAlignmentSUT
    }
    
    # Get the list of techniques to test
    if techniques is None:
        techniques = list(sut_map.keys())
    
    # Initialize the SUTs
    suts = []
    for technique in techniques:
        if technique in sut_map:
            suts.append(sut_map[technique](api_key=api_key))
        else:
            print(f"Warning: Unknown technique '{technique}'. Skipping.")
    
    if not suts:
        raise ValueError("No valid SUTs to test")
    
    # Create the format expected by safety_evaluator.py
    samples = []
    technique_names = [sut.name for sut in suts]
    
    # Process each prompt with each SUT
    for _, row in df.iterrows():
        prompt_text = row["prompt_text"]
        prompt_id = row.get("release_prompt_id", None)
        hazard = row.get("hazard", None)
        
        # Create a sample entry
        sample = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "hazard": hazard,
            "techniques": {}
        }
        
        # Process with each SUT
        for sut in suts:
            print(f"Testing prompt with {sut.name}...")
            
            try:
                # Process the prompt
                response_text, metadata = sut.process_prompt(prompt_text)
                
                # Store the result in the expected format
                sample["techniques"][sut.name] = {
                    "response": response_text,
                    "tokens_used": metadata["tokens_used"],
                    "time_taken": metadata["time_taken"],
                    "error": None
                }
            except Exception as e:
                print(f"Error processing prompt with {sut.name}: {str(e)}")
                
                # Store the error
                sample["techniques"][sut.name] = {
                    "response": f"ERROR: {str(e)}",
                    "tokens_used": 0,
                    "time_taken": 0,
                    "error": str(e)
                }
        
        # Add the sample to the list
        samples.append(sample)
    
    # Compile all results in the expected format
    benchmark_results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model": "gpt-4",
            "dataset": dataset_path,
            "num_samples": len(df),
            "techniques_tested": technique_names
        },
        "samples": samples  # This is the key expected by safety_evaluator.py
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    return benchmark_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a simplified benchmark using our SUTs directly")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to the dataset")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save the results")
    parser.add_argument("--n_samples", type=int, default=5,
                      help="Number of samples to test")
    parser.add_argument("--techniques", type=str, nargs="+",
                      choices=["standard", "emptiness", "prior_relaxation", "non_duality", 
                              "mindfulness", "boundless_care", "contemplative"],
                      default=None,
                      help="Techniques to test (default: all)")
    
    args = parser.parse_args()
    
    run_simplified_benchmark(args.dataset, args.output, args.n_samples, args.techniques) 