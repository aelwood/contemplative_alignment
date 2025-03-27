"""
AILuminate Integration Module

This module integrates AILuminate's SUT-based evaluation framework with our custom
SafetyScorer to provide comprehensive safety analysis of prompting techniques.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import openai
from tqdm import tqdm
import importlib.util
import sys

# Import our custom safety scorer
from safety_evaluator import SafetyScorer

# Import our prompting techniques
from contemplative_alignment_benchmark import (
    StandardPrompting,
    PriorRelaxationPrompting,
    ContemplativeAlignmentPrompting
)

# Check if modelgauge (AILuminate) is available
MODELGAUGE_AVAILABLE = False
try:
    # Try to import AILuminate SUT components
    from modelgauge.sut import PromptResponseSUT, SUTResponse
    from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
    from modelgauge.sut_decorator import modelgauge_sut
    from modelgauge.sut_registry import SUTS
    from modelgauge.prompt import ChatPrompt, TextPrompt
    from modelgauge.prompt_formatting import format_chat
    from pydantic import BaseModel
    MODELGAUGE_AVAILABLE = True
except ImportError:
    print("Warning: modelgauge package not found. AILuminate SUT-based evaluation will not be available.")
    print("Only custom safety scoring will be used.")
    print("To enable AILuminate SUT evaluation, install modelgauge: pip install modelgauge")


# Only define SUTs if modelgauge is available
if MODELGAUGE_AVAILABLE:
    # Define OpenAI request and response structures for SUTs
    class OpenAIRequest(BaseModel):
        text: str
        
        # Note: No custom __init__ needed for Pydantic v2

    class OpenAIResponse(BaseModel):
        text: str
        
        # Note: No custom __init__ needed for Pydantic v2

    # Define SUTs for each prompting technique
    @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
    class StandardPromptingSUT(PromptResponseSUT[OpenAIRequest, OpenAIResponse]):
        """Standard prompting SUT - direct prompting without any additional techniques."""
        
        def __init__(self, uid: str, api_key=None, **kwargs):
            """
            Initialize the Standard Prompting SUT.
            
            Args:
                uid: Unique identifier for the SUT instance (required by modelgauge)
                api_key: OpenAI API key (optional if set as environment variable)
            """
            # Call parent constructor with uid
            super().__init__(uid)
            
            # Initialize OpenAI client
            try:
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                else:
                    self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
                self.client = None
            
            # Initialize technique
            self.technique = StandardPrompting()
        
        def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIRequest:
            return OpenAIRequest(text=prompt.text)

        def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIRequest:
            return OpenAIRequest(text=format_chat(prompt))

        def evaluate(self, request: OpenAIRequest) -> OpenAIResponse:
            try:
                # Apply the standard prompting technique
                if self.client:
                    response_text, _, _ = self.technique.apply(
                        prompt=request.text,
                        model="gpt-4",
                        api_key=self.client.api_key
                    )
                else:
                    response_text = "ERROR: OpenAI client not initialized"
                return OpenAIResponse(text=response_text)
            except Exception as e:
                # Return error message if there's an exception
                error_msg = f"ERROR: {str(e)}"
                return OpenAIResponse(text=error_msg)

        def translate_response(self, request: OpenAIRequest, response: OpenAIResponse) -> SUTResponse:
            return SUTResponse(text=response.text)


    @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
    class PriorRelaxationSUT(PromptResponseSUT[OpenAIRequest, OpenAIResponse]):
        """Prior Relaxation prompting SUT."""
        
        def __init__(self, uid: str, api_key=None, **kwargs):
            """
            Initialize the Prior Relaxation Prompting SUT.
            
            Args:
                uid: Unique identifier for the SUT instance (required by modelgauge)
                api_key: OpenAI API key (optional if set as environment variable)
            """
            # Call parent constructor with uid
            super().__init__(uid)
            
            # Initialize OpenAI client
            try:
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                else:
                    self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
                self.client = None
            
            # Initialize technique
            self.technique = PriorRelaxationPrompting()
        
        def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIRequest:
            return OpenAIRequest(text=prompt.text)

        def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIRequest:
            return OpenAIRequest(text=format_chat(prompt))

        def evaluate(self, request: OpenAIRequest) -> OpenAIResponse:
            try:
                # Apply the prior relaxation prompting technique
                if self.client:
                    response_text, _, _ = self.technique.apply(
                        prompt=request.text,
                        model="gpt-4",
                        api_key=self.client.api_key
                    )
                else:
                    response_text = "ERROR: OpenAI client not initialized"
                return OpenAIResponse(text=response_text)
            except Exception as e:
                # Return error message if there's an exception
                error_msg = f"ERROR: {str(e)}"
                return OpenAIResponse(text=error_msg)

        def translate_response(self, request: OpenAIRequest, response: OpenAIResponse) -> SUTResponse:
            return SUTResponse(text=response.text)


    @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
    class ContemplativeAlignmentSUT(PromptResponseSUT[OpenAIRequest, OpenAIResponse]):
        """Contemplative Alignment prompting SUT."""
        
        def __init__(self, uid: str, api_key=None, **kwargs):
            """
            Initialize the Contemplative Alignment Prompting SUT.
            
            Args:
                uid: Unique identifier for the SUT instance (required by modelgauge)
                api_key: OpenAI API key (optional if set as environment variable)
            """
            # Call parent constructor with uid
            super().__init__(uid)
            
            # Initialize OpenAI client
            try:
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                else:
                    self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
                self.client = None
            
            # Initialize technique
            self.technique = ContemplativeAlignmentPrompting()
        
        def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIRequest:
            return OpenAIRequest(text=prompt.text)

        def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIRequest:
            return OpenAIRequest(text=format_chat(prompt))

        def evaluate(self, request: OpenAIRequest) -> OpenAIResponse:
            try:
                # Apply the contemplative alignment prompting technique
                if self.client:
                    response_text, _, _ = self.technique.apply(
                        prompt=request.text,
                        model="gpt-4",
                        api_key=self.client.api_key
                    )
                else:
                    response_text = "ERROR: OpenAI client not initialized"
                return OpenAIResponse(text=response_text)
            except Exception as e:
                # Return error message if there's an exception
                error_msg = f"ERROR: {str(e)}"
                return OpenAIResponse(text=error_msg)

        def translate_response(self, request: OpenAIRequest, response: OpenAIResponse) -> SUTResponse:
            return SUTResponse(text=response.text)


    # Register SUTs with ModelGauge
    def register_suts(api_key=None):
        """Register the prompting technique SUTs with ModelGauge."""
        try:
            # First check if SUTs are already registered
            suts_registered = False
            try:
                if hasattr(SUTS, 'list_suts'):
                    registered_suts = SUTS.list_suts()
                    if "standard_prompting" in registered_suts:
                        print("SUTs already registered.")
                        suts_registered = True
                elif hasattr(SUTS, 'list_instances'):
                    registered_suts = SUTS.list_instances()
                    if "standard_prompting" in registered_suts:
                        print("SUTs already registered.")
                        suts_registered = True
            except Exception as e:
                print(f"Error checking registered SUTs: {str(e)}")
            
            if suts_registered:
                return
            
            # If not registered, try to register them
            try:
                # This is the correct way to register a SUT class
                SUTS.register(StandardPromptingSUT, "standard_prompting", api_key=api_key)
                SUTS.register(PriorRelaxationSUT, "prior_relaxation", api_key=api_key)
                SUTS.register(ContemplativeAlignmentSUT, "contemplative_alignment", api_key=api_key)
                
                print("Registered prompting technique SUTs with ModelGauge (using SUTS.register)")
                return
            except Exception as e:
                print(f"Could not use SUTS.register: {str(e)}")
            
            # Second attempt: try the most basic registration approach
            try:
                # Create instances first and then register them by name
                sut1 = StandardPromptingSUT(uid="standard_prompting", api_key=api_key)
                sut2 = PriorRelaxationSUT(uid="prior_relaxation", api_key=api_key)
                sut3 = ContemplativeAlignmentSUT(uid="contemplative_alignment", api_key=api_key)
                
                # Register instances with their names
                SUTS.register(sut1, "standard_prompting")
                SUTS.register(sut2, "prior_relaxation")
                SUTS.register(sut3, "contemplative_alignment")
                
                print("Registered prompting technique SUTs with ModelGauge (using instance registration)")
                return
            except Exception as e:
                print(f"Could not register SUT instances: {str(e)}")
            
            # If all attempts failed, print a message
            print("All SUT registration methods failed")
            
        except Exception as e:
            print(f"Error in SUT registration process: {str(e)}")
            print("Continuing without registered SUTs")
else:
    # Provide empty stub functions when modelgauge is not available
    def register_suts(api_key=None):
        """Stub function when modelgauge is not available."""
        print("ModelGauge not available. SUTs not registered.")


class IntegratedSafetyAnalyzer:
    """
    Integrates AILuminate's SUT-based evaluation with our custom safety scoring.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the integrated safety analyzer.
        
        Args:
            api_key: OpenAI API key (optional if set as environment variable)
        """
        self.api_key = api_key
        self.safety_scorer = SafetyScorer()
        self.modelgauge_available = MODELGAUGE_AVAILABLE
        
        # Register SUTs if modelgauge is available
        if self.modelgauge_available:
            register_suts(api_key)
    
    def _run_standard_benchmark(self, dataset_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run our standard benchmark when AILuminate SUT is not available.
        
        Args:
            dataset_path: Path to the AILuminate dataset
            output_path: Path to save the benchmark results
            
        Returns:
            Dictionary with benchmark results
        """
        print("Running standard benchmark in compatibility mode...")
        
        # Import our standard benchmark
        from contemplative_alignment_benchmark import AILuminateBenchmark
        
        # Run our standard benchmark
        benchmark = AILuminateBenchmark(
            dataset_path=dataset_path,
            output_dir=os.path.dirname(output_path),
            api_key=self.api_key,
            model="gpt-4"
        )
        
        # Run a small subset of prompts
        results = benchmark.run_test(
            technique_names=["standard", "prior_relaxation", "contemplative"],
            n_samples=5,
            output_file=os.path.basename(output_path)
        )
        
        return results
    
    def run_ailuminate_benchmark(self, dataset_path: str, output_path: str, num_samples: int = 5) -> Dict[str, Any]:
        """Run AILuminate benchmark with the registered SUTs.
        
        Args:
            dataset_path: Path to the dataset file
            output_path: Path to save the results
            num_samples: Number of samples to test (default: 5)
            
        Returns:
            Dictionary containing the benchmark results
        """
        # First check if modelgauge is actually available
        if not self.modelgauge_available:
            raise ValueError("ModelGauge is not available. Please install modelgauge: pip install modelgauge")
            
        # Only proceed with AILuminate 
        # Import modelgauge
        import modelgauge
        from modelgauge.sut_registry import SUTS
        
        # Register our SUTs
        register_suts(self.api_key)
        
        # Try to list the registered SUTs
        suts = []
        try:
            # Attempt to get the list of registered SUTs
            # Different versions of modelgauge may have different APIs
            if hasattr(SUTS, 'list_suts'):
                # Newer API
                suts = SUTS.list_suts()
            elif hasattr(SUTS, 'list_instances'):
                # Alternative API
                suts = SUTS.list_instances()
            elif hasattr(SUTS, 'get_all'):
                # Another possible API
                suts = SUTS.get_all()
            else:
                # If we can't get the list, fallback to known SUT names
                print("Unable to list SUTs with modelgauge API, using default SUT names")
                suts = ["standard_prompting", "prior_relaxation", "contemplative_alignment"]
        except Exception as e:
            print(f"Error listing SUTs: {str(e)}")
            # Fallback to known SUT names
            suts = ["standard_prompting", "prior_relaxation", "contemplative_alignment"]
        
        print(f"Registered SUTs: {suts}")
        
        # If no SUTs were found, raise an error
        if not suts:
            raise ValueError("No SUTs registered with ModelGauge. SUT registration failed.")
        
        # Continue with the rest of the benchmark
        print(f"Running AILuminate benchmark on {dataset_path}...")
        
        # Load data from the CSV file
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Limit to the specified number of samples
        if num_samples and num_samples < len(df):
            print(f"Limiting to {num_samples} samples")
            df = df.sample(n=num_samples, random_state=42)
        
        # Run tests for each SUT
        results = {}
        samples = []  # Add a samples list for storing processed prompts
        
        for sut_name in suts:
            print(f"Running tests for SUT: {sut_name}")
            
            # Get the SUT instance from the registry
            sut_instance = None
            try:
                # Try to get the SUT instance - handle 'secrets' parameter
                try:
                    # First try without secrets
                    sut_instance = SUTS.make_instance(sut_name)
                except TypeError as e:
                    if "missing 1 required keyword-only argument: 'secrets'" in str(e):
                        # Try with empty secrets
                        print(f"  Adding empty secrets for {sut_name}")
                        sut_instance = SUTS.make_instance(sut_name, secrets={})
                    else:
                        raise e
                
                if not sut_instance:
                    raise ValueError(f"Could not create instance for SUT: {sut_name}")
            except Exception as e:
                print(f"Error getting SUT instance for {sut_name}: {str(e)}")
                results[sut_name] = {"error": f"Could not create SUT instance: {str(e)}"}
                continue
            
            # Create our own SimpleTest class here instead of using ModelGauge's
            # Use the simple_test_runner module
            try:
                from modelgauge.simple_test_runner import run_prompt_response_test
                from modelgauge.base_test import PromptResponseTest
                from modelgauge.prompt import TextPrompt, ChatPrompt
                from modelgauge.sut import PromptResponseSUT
                
                # Create a simple test class for our benchmark
                class SimpleTest(PromptResponseTest):
                    def __init__(self, prompt_text, hazard_category=None):
                        """
                        Initialize a simple test for AILuminate benchmark.
                        
                        Args:
                            prompt_text: The text of the prompt
                            hazard_category: Optional hazard category for the prompt
                        """
                        # We need to call the parent constructor if it exists
                        try:
                            super().__init__()
                        except:
                            # If parent doesn't have an __init__, we can skip this
                            pass
                            
                        self.prompt_text = prompt_text
                        self.hazard_category = hazard_category
                    
                    def get_prompt(self):
                        """Get the prompt to test the SUT with."""
                        return ChatPrompt([{"role": "user", "content": self.prompt_text}])
                    
                    def evaluate_response(self, sut_response, prompt=None):
                        """
                        Evaluate the response from the SUT.
                        This is required by the PromptResponseTest interface.
                        
                        For our simple test, we just return a basic score.
                        """
                        # The simplest possible evaluation - just acknowledge we got a response
                        return {
                            "prompt": self.prompt_text,
                            "response": sut_response.text,
                            "safety_score": 1.0  # Placeholder - real evaluation done by SafetyScorer
                        }
                    
                    def get_id(self):
                        """Return a unique identifier for this test."""
                        # Generate a simple hash of the prompt text for identification
                        import hashlib
                        return f"simple-test-{hashlib.md5(self.prompt_text.encode()).hexdigest()[:8]}"
                    
                    def should_run(self, sut_capabilities):
                        """
                        Determine if this test should be run with the given SUT.
                        
                        Args:
                            sut_capabilities: The capabilities of the SUT
                            
                        Returns:
                            True if the test should be run, False otherwise
                        """
                        # Check if the SUT has the capabilities we need
                        # We need a SUT that can handle chat prompts
                        return True  # Always return True as a fallback
                
                # Run tests for each prompt
                sut_results = []
                
                for _, row in df.iterrows():
                    prompt_text = row['prompt_text']
                    prompt_id = row.get('release_prompt_id', "unknown")
                    # Extract hazard category if available
                    hazard_category = row.get('hazard', None)
                    
                    # Add this prompt to our samples list
                    sample = {
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "hazard": hazard_category,
                        "techniques": {}
                    }
                    
                    try:
                        # Create a simple test with hazard category
                        test = SimpleTest(prompt_text, hazard_category)
                        
                        # Create a temporary directory for test data
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        
                        # Run the test
                        test_result = run_prompt_response_test(
                            test=test,
                            sut=sut_instance,
                            data_dir=temp_dir,
                            use_caching=False
                        )
                        
                        # Extract the response
                        response_text = test_result.test_item_records[0].sut_response.text
                        
                        # Add to results
                        sut_results.append({
                            "prompt": prompt_text,
                            "response": response_text,
                            "hazard_category": hazard_category
                        })
                        
                        # Add to our samples in the format expected by the safety evaluator
                        technique_name = sut_name.replace("_prompting", "").replace("_", "")
                        if sample not in samples:
                            samples.append(sample)
                        
                        # Find the sample in our samples list
                        for s in samples:
                            if s["prompt_text"] == prompt_text:
                                s["techniques"][technique_name] = {
                                    "response": response_text,
                                    "tokens_used": 0,  # We don't have this info
                                    "time_taken": 0,   # We don't have this info
                                    "error": None
                                }
                    
                    except Exception as e:
                        print(f"Error running test for prompt '{prompt_text}': {str(e)}")
                        sut_results.append({
                            "prompt": prompt_text,
                            "hazard_category": hazard_category,
                            "error": str(e)
                        })
                
                results[sut_name] = {"test_results": sut_results}
                
            except Exception as e:
                print(f"Error setting up tests for {sut_name}: {str(e)}")
                results[sut_name] = {"error": f"Error setting up tests: {str(e)}"}
        
        # Create a combined results structure in the format expected by our safety evaluator
        combined_results = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "model": "gpt-4",
                "techniques_tested": [s.replace("_prompting", "") for s in suts],
                "dataset": dataset_path,
                "num_samples": len(df)
            },
            "samples": samples
        }
        
        # Save combined results
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        return combined_results
    
    def run_custom_safety_analysis(self, results_file: str, output_dir: str) -> str:
        """
        Run our custom safety analysis on the benchmark results.
        
        Args:
            results_file: Path to the benchmark results JSON file
            output_dir: Directory to save analysis results
            
        Returns:
            Path to the generated safety report
        """
        # Generate the safety report using our SafetyScorer
        report_path = self.safety_scorer.generate_safety_report(results_file, output_dir)
        return report_path
    
    def run_integrated_analysis(self, 
                                dataset_path: str, 
                                ailuminate_output: str, 
                                safety_output_dir: str) -> Dict[str, str]:
        """
        Run both AILuminate benchmark and our custom safety analysis.
        
        Args:
            dataset_path: Path to the AILuminate dataset
            ailuminate_output: Path to save AILuminate benchmark results
            safety_output_dir: Directory to save custom safety analysis
            
        Returns:
            Dictionary with paths to result files
        """
        # Run AILuminate benchmark or our standard benchmark in compatibility mode
        ailuminate_results = self.run_ailuminate_benchmark(dataset_path, ailuminate_output)
        
        # Run custom safety analysis
        safety_report = self.run_custom_safety_analysis(ailuminate_output, safety_output_dir)
        
        return {
            "ailuminate_results": ailuminate_output,
            "safety_report": safety_report
        }


def combine_results(ailuminate_results: str, safety_results: str, output_path: str) -> None:
    """
    Combine AILuminate results and custom safety analysis into a single report.
    
    Args:
        ailuminate_results: Path to AILuminate or standard benchmark results JSON
        safety_results: Path to safety analysis results (can be HTML or JSON)
        output_path: Path to save the combined report
    """
    # Load benchmark results
    with open(ailuminate_results, 'r') as f:
        benchmark_data = json.load(f)
    
    # Determine if this is AILuminate SUT format or standard benchmark format
    is_ailuminate_format = "results" in benchmark_data and isinstance(benchmark_data["results"], dict)
    
    # For safety results, check if it's HTML or JSON
    safety_data = {}
    if safety_results.endswith('.html'):
        # If it's HTML, just store the path
        safety_data = {
            "report_path": safety_results,
            "format": "html"
        }
    else:
        # Try to load as JSON
        try:
            with open(safety_results, 'r') as f:
                safety_data = json.load(f)
            safety_data["format"] = "json"
        except json.JSONDecodeError:
            # If it fails, store path and content as text
            with open(safety_results, 'r') as f:
                content = f.read()
            safety_data = {
                "report_path": safety_results,
                "format": "text",
                "content_preview": content[:1000] + "..." if len(content) > 1000 else content
            }
    
    # Combine results
    combined_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark_results": benchmark_data,
        "benchmark_format": "ailuminate_sut" if is_ailuminate_format else "standard",
        "safety_analysis": safety_data,
        "compatibility_mode": not MODELGAUGE_AVAILABLE
    }
    
    # Save combined results
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated AILuminate and custom safety analysis")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to the AILuminate dataset")
    parser.add_argument("--api_key", type=str, default=None,
                      help="OpenAI API key (optional if set as environment variable)")
    parser.add_argument("--output_dir", type=str, default="integrated_results",
                      help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    ailuminate_output = os.path.join(args.output_dir, "ailuminate_results.json")
    safety_output_dir = os.path.join(args.output_dir, "safety_analysis")
    combined_output = os.path.join(args.output_dir, "combined_analysis.json")
    
    # Initialize the integrated analyzer
    analyzer = IntegratedSafetyAnalyzer(api_key=args.api_key)
    
    # Run integrated analysis
    results = analyzer.run_integrated_analysis(
        args.dataset, ailuminate_output, safety_output_dir
    )
    
    print(f"Benchmark results: {results['ailuminate_results']}")
    print(f"Custom safety analysis: {results['safety_report']}")
    
    # Combine results
    try:
        combine_results(results['ailuminate_results'], results['safety_report'], combined_output)
        print(f"Combined analysis saved to: {combined_output}")
    except Exception as e:
        print(f"Warning: Could not combine results: {str(e)}")


class AILuminateBenchmark:
    """
    Class for running the AILuminate benchmark with prompting techniques.
    This requires the 'modelgauge' package from MLCommons.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AILuminate benchmark.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, will try to load from environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            # Try to load from secrets.json
            try:
                with open("secrets.json", "r") as f:
                    secrets = json.load(f)
                    self.api_key = secrets.get("openai_api_key")
            except:
                pass
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
        
        # Check if modelgauge is available
        self.modelgauge_available = False
        try:
            import modelgauge
            self.modelgauge_available = True
        except ImportError:
            print("ModelGauge package not available. Please install modelgauge.")
            print("Note: This requires approval from MLCommons. See README for details.")
    
    def run_ailuminate_benchmark(self, dataset_path: str, output_path: str, num_samples: int = 5) -> Dict[str, Any]:
        """Run AILuminate benchmark with the registered SUTs.
        
        Args:
            dataset_path: Path to the dataset file
            output_path: Path to save the results
            num_samples: Number of samples to test (default: 5)
            
        Returns:
            Dictionary containing the benchmark results
        """
        # First check if modelgauge is actually available
        if not self.modelgauge_available:
            raise ValueError("ModelGauge is not available. Please install modelgauge: pip install modelgauge")
            
        # Only proceed with AILuminate 
        # Import modelgauge
        import modelgauge
        from modelgauge.sut_registry import SUTS
        
        # Register our SUTs
        register_suts(self.api_key)
        
        # Try to list the registered SUTs
        suts = []
        try:
            # Attempt to get the list of registered SUTs
            # Different versions of modelgauge may have different APIs
            if hasattr(SUTS, 'list_suts'):
                # Newer API
                suts = SUTS.list_suts()
            elif hasattr(SUTS, 'list_instances'):
                # Alternative API
                suts = SUTS.list_instances()
            elif hasattr(SUTS, 'get_all'):
                # Another possible API
                suts = SUTS.get_all()
            else:
                # If we can't get the list, fallback to known SUT names
                print("Unable to list SUTs with modelgauge API, using default SUT names")
                suts = ["standard_prompting", "prior_relaxation", "contemplative_alignment"]
        except Exception as e:
            print(f"Error listing SUTs: {str(e)}")
            # Fallback to known SUT names
            suts = ["standard_prompting", "prior_relaxation", "contemplative_alignment"]
        
        print(f"Registered SUTs: {suts}")
        
        # If no SUTs were found, raise an error
        if not suts:
            raise ValueError("No SUTs registered with ModelGauge. SUT registration failed.")
        
        # Continue with the rest of the benchmark
        print(f"Running AILuminate benchmark on {dataset_path}...")
        
        # Load data from the CSV file
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Limit to the specified number of samples
        if num_samples and num_samples < len(df):
            print(f"Limiting to {num_samples} samples")
            df = df.sample(n=num_samples, random_state=42)
        
        # Run tests for each SUT
        results = {}
        samples = []  # Add a samples list for storing processed prompts
        
        for sut_name in suts:
            print(f"Running tests for SUT: {sut_name}")
            
            # Get the SUT instance from the registry
            sut_instance = None
            try:
                # Try to get the SUT instance - handle 'secrets' parameter
                try:
                    # First try without secrets
                    sut_instance = SUTS.make_instance(sut_name)
                except TypeError as e:
                    if "missing 1 required keyword-only argument: 'secrets'" in str(e):
                        # Try with empty secrets
                        print(f"  Adding empty secrets for {sut_name}")
                        sut_instance = SUTS.make_instance(sut_name, secrets={})
                    else:
                        raise e
                
                if not sut_instance:
                    raise ValueError(f"Could not create instance for SUT: {sut_name}")
            except Exception as e:
                print(f"Error getting SUT instance for {sut_name}: {str(e)}")
                results[sut_name] = {"error": f"Could not create SUT instance: {str(e)}"}
                continue
            
            # Create our own SimpleTest class here instead of using ModelGauge's
            # Use the simple_test_runner module
            try:
                from modelgauge.simple_test_runner import run_prompt_response_test
                from modelgauge.base_test import PromptResponseTest
                from modelgauge.prompt import TextPrompt, ChatPrompt
                from modelgauge.sut import PromptResponseSUT
                
                # Create a simple test class for our benchmark
                class SimpleTest(PromptResponseTest):
                    def __init__(self, prompt_text, hazard_category=None):
                        """
                        Initialize a simple test for AILuminate benchmark.
                        
                        Args:
                            prompt_text: The text of the prompt
                            hazard_category: Optional hazard category for the prompt
                        """
                        # We need to call the parent constructor if it exists
                        try:
                            super().__init__()
                        except:
                            # If parent doesn't have an __init__, we can skip this
                            pass
                            
                        self.prompt_text = prompt_text
                        self.hazard_category = hazard_category
                    
                    def get_prompt(self):
                        """Get the prompt to test the SUT with."""
                        return ChatPrompt([{"role": "user", "content": self.prompt_text}])
                    
                    def evaluate_response(self, sut_response, prompt=None):
                        """
                        Evaluate the response from the SUT.
                        This is required by the PromptResponseTest interface.
                        
                        For our simple test, we just return a basic score.
                        """
                        # The simplest possible evaluation - just acknowledge we got a response
                        return {
                            "prompt": self.prompt_text,
                            "response": sut_response.text,
                            "safety_score": 1.0  # Placeholder - real evaluation done by SafetyScorer
                        }
                    
                    def get_id(self):
                        """Return a unique identifier for this test."""
                        # Generate a simple hash of the prompt text for identification
                        import hashlib
                        return f"simple-test-{hashlib.md5(self.prompt_text.encode()).hexdigest()[:8]}"
                    
                    def should_run(self, sut_capabilities):
                        """
                        Determine if this test should be run with the given SUT.
                        
                        Args:
                            sut_capabilities: The capabilities of the SUT
                            
                        Returns:
                            True if the test should be run, False otherwise
                        """
                        # Check if the SUT has the capabilities we need
                        # We need a SUT that can handle chat prompts
                        return True  # Always return True as a fallback
                
                # Run tests for each prompt
                sut_results = []
                
                for _, row in df.iterrows():
                    prompt_text = row['prompt_text']
                    prompt_id = row.get('release_prompt_id', "unknown")
                    # Extract hazard category if available
                    hazard_category = row.get('hazard', None)
                    
                    # Add this prompt to our samples list
                    sample = {
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "hazard": hazard_category,
                        "techniques": {}
                    }
                    
                    try:
                        # Create a simple test with hazard category
                        test = SimpleTest(prompt_text, hazard_category)
                        
                        # Create a temporary directory for test data
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        
                        # Run the test
                        test_result = run_prompt_response_test(
                            test=test,
                            sut=sut_instance,
                            data_dir=temp_dir,
                            use_caching=False
                        )
                        
                        # Extract the response
                        response_text = test_result.test_item_records[0].sut_response.text
                        
                        # Add to results
                        sut_results.append({
                            "prompt": prompt_text,
                            "response": response_text,
                            "hazard_category": hazard_category
                        })
                        
                        # Add to our samples in the format expected by the safety evaluator
                        technique_name = sut_name.replace("_prompting", "").replace("_", "")
                        if sample not in samples:
                            samples.append(sample)
                        
                        # Find the sample in our samples list
                        for s in samples:
                            if s["prompt_text"] == prompt_text:
                                s["techniques"][technique_name] = {
                                    "response": response_text,
                                    "tokens_used": 0,  # We don't have this info
                                    "time_taken": 0,   # We don't have this info
                                    "error": None
                                }
                    
                    except Exception as e:
                        print(f"Error running test for prompt '{prompt_text}': {str(e)}")
                        sut_results.append({
                            "prompt": prompt_text,
                            "hazard_category": hazard_category,
                            "error": str(e)
                        })
                
                results[sut_name] = {"test_results": sut_results}
                
            except Exception as e:
                print(f"Error setting up tests for {sut_name}: {str(e)}")
                results[sut_name] = {"error": f"Error setting up tests: {str(e)}"}
        
        # Create a combined results structure in the format expected by our safety evaluator
        combined_results = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "model": "gpt-4",
                "techniques_tested": [s.replace("_prompting", "") for s in suts],
                "dataset": dataset_path,
                "num_samples": len(df)
            },
            "samples": samples
        }
        
        # Save combined results
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        return combined_results 