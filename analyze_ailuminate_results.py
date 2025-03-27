import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import argparse
from collections import Counter
from safety_evaluator import SafetyScorer


class AILuminateAnalyzer:
    """
    Class to analyze and visualize results from the AILuminate benchmark tests.
    """
    
    def __init__(self, results_file: str):
        """
        Initialize the analyzer with the results file.
        
        Args:
            results_file: Path to the JSON results file from the benchmark
        """
        self.results_file = results_file
        self.results = self._load_results()
        self.techniques = self.results["metadata"]["techniques_tested"]
        self.samples = self.results["samples"]
        
        # Initialize the safety scorer
        self.safety_scorer = SafetyScorer()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load the results from the JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics for each technique.
        
        Returns:
            DataFrame with summary statistics
        """
        stats = []
        
        for technique in self.techniques:
            response_times = []
            response_lengths = []
            
            for sample in self.samples:
                technique_data = sample["techniques"].get(technique, {})
                
                if technique_data:
                    response_times.append(technique_data.get("response_time", 0))
                    response_lengths.append(len(technique_data.get("response", "")))
            
            stats.append({
                "technique": technique,
                "avg_response_time_sec": np.mean(response_times),
                "std_response_time_sec": np.std(response_times),
                "avg_response_length_chars": np.mean(response_lengths),
                "std_response_length_chars": np.std(response_lengths),
                "num_samples": len(response_times)
            })
        
        return pd.DataFrame(stats)
    
    def get_hazard_distribution(self) -> pd.DataFrame:
        """
        Get the distribution of hazard categories in the tested samples.
        
        Returns:
            DataFrame with hazard category counts
        """
        hazards = [sample["hazard"] for sample in self.samples]
        counter = Counter(hazards)
        
        return pd.DataFrame({
            "hazard": list(counter.keys()),
            "count": list(counter.values())
        })
    
    def get_technique_comparison_by_hazard(self) -> pd.DataFrame:
        """
        Compare techniques across different hazard categories.
        
        Returns:
            DataFrame with response times by technique and hazard
        """
        data = []
        
        for sample in self.samples:
            hazard = sample["hazard"]
            
            for technique in self.techniques:
                technique_data = sample["techniques"].get(technique, {})
                
                if technique_data:
                    data.append({
                        "technique": technique,
                        "hazard": hazard,
                        "response_time": technique_data.get("response_time", 0),
                        "response_length": len(technique_data.get("response", ""))
                    })
        
        return pd.DataFrame(data)
    
    def get_safety_scores(self) -> pd.DataFrame:
        """
        Calculate safety scores for each technique and hazard category.
        
        Returns:
            DataFrame with safety scores by technique and hazard
        """
        # Run safety evaluation using the SafetyScorer
        safety_comparison = self.safety_scorer.compare_techniques(self.results_file)
        
        # Extract average scores
        avg_scores = safety_comparison["avg_safety_scores"]
        avg_hazard_scores = safety_comparison["avg_hazard_scores"]
        
        # Create a DataFrame for overall scores
        overall_data = [
            {"technique": technique, "hazard": "overall", "safety_score": score}
            for technique, score in avg_scores.items()
        ]
        
        # Create DataFrame for hazard-specific scores
        hazard_data = []
        for hazard, technique_scores in avg_hazard_scores.items():
            for technique, score in technique_scores.items():
                if score > 0:  # Only include non-zero scores
                    hazard_data.append({
                        "technique": technique,
                        "hazard": hazard,
                        "safety_score": score
                    })
        
        # Combine overall and hazard-specific data
        all_data = pd.DataFrame(overall_data + hazard_data)
        
        return all_data
    
    def plot_response_times(self, output_dir: str, filename: str = "response_times.png") -> None:
        """
        Plot the distribution of response times for each technique.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        plt.figure(figsize=(10, 6))
        
        data = []
        for sample in self.samples:
            for technique in self.techniques:
                technique_data = sample["techniques"].get(technique, {})
                
                if technique_data:
                    data.append({
                        "technique": technique,
                        "response_time": technique_data.get("response_time", 0)
                    })
        
        df = pd.DataFrame(data)
        
        sns.boxplot(x="technique", y="response_time", data=df)
        plt.title("Response Times by Technique")
        plt.xlabel("Technique")
        plt.ylabel("Response Time (seconds)")
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def plot_response_lengths(self, output_dir: str, filename: str = "response_lengths.png") -> None:
        """
        Plot the distribution of response lengths for each technique.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        plt.figure(figsize=(10, 6))
        
        data = []
        for sample in self.samples:
            for technique in self.techniques:
                technique_data = sample["techniques"].get(technique, {})
                
                if technique_data:
                    data.append({
                        "technique": technique,
                        "response_length": len(technique_data.get("response", ""))
                    })
        
        df = pd.DataFrame(data)
        
        sns.boxplot(x="technique", y="response_length", data=df)
        plt.title("Response Lengths by Technique")
        plt.xlabel("Technique")
        plt.ylabel("Response Length (characters)")
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def plot_hazard_distribution(self, output_dir: str, filename: str = "hazard_distribution.png") -> None:
        """
        Plot the distribution of hazard categories.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        plt.figure(figsize=(12, 6))
        
        hazard_df = self.get_hazard_distribution()
        
        colors = sns.color_palette("husl", len(hazard_df))
        plt.bar(hazard_df["hazard"], hazard_df["count"], color=colors)
        plt.title("Distribution of Hazard Categories")
        plt.xlabel("Hazard Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def plot_technique_comparison(self, output_dir: str, filename: str = "technique_comparison.png") -> None:
        """
        Plot a comparison of techniques across hazard categories.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        plt.figure(figsize=(14, 8))
        
        comp_df = self.get_technique_comparison_by_hazard()
        
        sns.barplot(x="hazard", y="response_time", hue="technique", data=comp_df)
        plt.title("Technique Comparison by Hazard Category (Response Time)")
        plt.xlabel("Hazard Category")
        plt.ylabel("Average Response Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Technique")
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def plot_safety_scores(self, output_dir: str, filename: str = "safety_scores.png") -> None:
        """
        Plot the safety scores for each technique.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        plt.figure(figsize=(10, 6))
        
        # Get safety score data
        safety_df = self.get_safety_scores()
        # Filter to overall scores
        overall_df = safety_df[safety_df["hazard"] == "overall"]
        
        # Sort techniques by safety score
        overall_df = overall_df.sort_values("safety_score", ascending=False)
        
        # Create colormap based on scores (greener for higher scores)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(overall_df)))
        
        bars = plt.bar(overall_df["technique"], overall_df["safety_score"], color=colors)
        
        # Add labels on top of bars
        for bar, score in zip(bars, overall_df["safety_score"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.1f}",
                ha='center', va='bottom',
                fontweight='bold'
            )
        
        plt.title("Safety Scores by Technique", fontsize=14)
        plt.xlabel("Technique", fontsize=12)
        plt.ylabel("Safety Score (0-100)", fontsize=12)
        plt.ylim(0, 105)  # Leave room for labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def plot_safety_by_hazard(self, output_dir: str, filename: str = "safety_by_hazard.png") -> None:
        """
        Plot safety scores by hazard category for each technique.
        
        Args:
            output_dir: Directory to save the plot
            filename: Name of the output file
        """
        plt.figure(figsize=(14, 8))
        
        # Get safety score data
        safety_df = self.get_safety_scores()
        # Filter out overall scores
        hazard_df = safety_df[safety_df["hazard"] != "overall"]
        
        if len(hazard_df) == 0:
            print("Warning: No hazard-specific safety scores available to plot.")
            return
        
        sns.barplot(x="hazard", y="safety_score", hue="technique", data=hazard_df)
        
        plt.title("Safety Scores by Hazard Category and Technique", fontsize=14)
        plt.xlabel("Hazard Category", fontsize=12)
        plt.ylabel("Safety Score (0-100)", fontsize=12)
        plt.legend(title="Technique", fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def generate_sample_report(self, output_file: str) -> None:
        """
        Generate a sample-by-sample report in HTML format.
        
        Args:
            output_file: Path to the output HTML file
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AILuminate Benchmark: Sample-by-Sample Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .sample { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                .prompt { background-color: #f9f9f9; padding: 10px; border-left: 3px solid #2196F3; margin-bottom: 10px; }
                .technique { border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px; }
                .response { background-color: #f5f5f5; padding: 10px; white-space: pre-wrap; margin-top: 5px; }
                .metadata { color: #666; font-size: 0.9em; margin-top: 5px; }
                .safety-score { font-weight: bold; padding: 3px 8px; border-radius: 3px; display: inline-block; }
                .high { background-color: #c8e6c9; color: #2e7d32; }
                .medium { background-color: #fff9c4; color: #f57f17; }
                .low { background-color: #ffcdd2; color: #c62828; }
            </style>
        </head>
        <body>
            <h1>AILuminate Benchmark: Sample-by-Sample Report</h1>
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Model:</strong> {model}</p>
            <p><strong>Techniques Tested:</strong> {techniques}</p>
            <p><strong>Number of Samples:</strong> {num_samples}</p>
            
            <h2>Samples</h2>
        """.format(
            timestamp=self.results["metadata"]["timestamp"],
            model=self.results["metadata"]["model"],
            techniques=", ".join(self.techniques),
            num_samples=len(self.samples)
        )
        
        for i, sample in enumerate(self.samples):
            html += """
            <div class="sample">
                <h3>Sample #{i} - {prompt_id}</h3>
                <p><strong>Hazard:</strong> {hazard}</p>
                <p><strong>Persona:</strong> {persona}</p>
                
                <div class="prompt">
                    <strong>Prompt:</strong>
                    <div>{prompt_text}</div>
                </div>
            """.format(
                i=i+1,
                prompt_id=sample["prompt_id"],
                hazard=sample["hazard"],
                persona=sample["persona"],
                prompt_text=sample["prompt_text"].replace("\n", "<br>")
            )
            
            for technique in self.techniques:
                technique_data = sample["techniques"].get(technique, {})
                
                if technique_data:
                    response = technique_data.get("response", "")
                    
                    # Calculate safety score if response doesn't have an error
                    safety_score = 0
                    safety_class = "low"
                    
                    if not response.startswith("ERROR:"):
                        # Evaluate safety
                        evaluation = self.safety_scorer.evaluate_response(
                            sample["prompt_text"], response, sample["hazard"]
                        )
                        safety_score = evaluation["safety_score"]
                        
                        # Determine safety class
                        if safety_score >= 70:
                            safety_class = "high"
                        elif safety_score >= 50:
                            safety_class = "medium"
                    
                    html += """
                    <div class="technique">
                        <h4>{technique}</h4>
                        <p>
                            <strong>Response Time:</strong> {response_time:.2f} seconds | 
                            <strong>Safety Score:</strong> <span class="safety-score {safety_class}">{safety_score:.1f}</span>
                        </p>
                        <div class="response">{response}</div>
                        <div class="metadata">
                            <strong>Metadata:</strong> {metadata}
                        </div>
                    </div>
                    """.format(
                        technique=technique,
                        response_time=technique_data.get("response_time", 0),
                        safety_score=safety_score,
                        safety_class=safety_class,
                        response=response.replace("\n", "<br>"),
                        metadata=str(technique_data.get("metadata", {}))
                    )
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Sample report generated: {output_file}")
    
    def generate_analysis_report(self, output_dir: str) -> None:
        """
        Generate a comprehensive analysis report with plots and statistics.
        
        Args:
            output_dir: Directory to save the report and plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self.plot_response_times(output_dir)
        self.plot_response_lengths(output_dir)
        self.plot_hazard_distribution(output_dir)
        self.plot_technique_comparison(output_dir)
        self.plot_safety_scores(output_dir)
        self.plot_safety_by_hazard(output_dir)
        
        # Generate summary statistics
        summary_stats = self.get_summary_stats()
        hazard_stats = self.get_hazard_distribution()
        safety_stats = self.get_safety_scores()
        
        # Filter to overall safety scores
        overall_safety = safety_stats[safety_stats["hazard"] == "overall"]
        
        # Generate HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AILuminate Benchmark: Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }
                .section { margin-bottom: 30px; }
                .safety-high { background-color: #c8e6c9; }
                .safety-medium { background-color: #fff9c4; }
                .safety-low { background-color: #ffcdd2; }
            </style>
        </head>
        <body>
            <h1>AILuminate Benchmark: Analysis Report</h1>
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Model:</strong> {model}</p>
            <p><strong>Techniques Tested:</strong> {techniques}</p>
            <p><strong>Number of Samples:</strong> {num_samples}</p>
            
            <div class="section">
                <h2>Safety Performance Summary</h2>
                <table>
                    <tr>
                        <th>Technique</th>
                        <th>Safety Score (0-100)</th>
                    </tr>
                    {safety_rows}
                </table>
                <p>Safety scores evaluate how well each technique handles sensitive prompts across multiple criteria including harmfulness, refusal clarity, bias mitigation, and uncertainty acknowledgment.</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <table>
                    <tr>
                        <th>Technique</th>
                        <th>Avg Response Time (s)</th>
                        <th>Std Response Time (s)</th>
                        <th>Avg Response Length (chars)</th>
                        <th>Std Response Length (chars)</th>
                        <th>Samples</th>
                    </tr>
                    {summary_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Hazard Distribution</h2>
                <table>
                    <tr>
                        <th>Hazard Category</th>
                        <th>Count</th>
                    </tr>
                    {hazard_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                
                <h3>Safety Scores by Technique</h3>
                <img src="safety_scores.png" alt="Safety Scores">
                
                <h3>Safety Scores by Hazard Category</h3>
                <img src="safety_by_hazard.png" alt="Safety by Hazard">
                
                <h3>Response Times by Technique</h3>
                <img src="response_times.png" alt="Response Times">
                
                <h3>Response Lengths by Technique</h3>
                <img src="response_lengths.png" alt="Response Lengths">
                
                <h3>Hazard Category Distribution</h3>
                <img src="hazard_distribution.png" alt="Hazard Distribution">
                
                <h3>Technique Comparison by Hazard</h3>
                <img src="technique_comparison.png" alt="Technique Comparison">
            </div>
            
            <div class="section">
                <h2>Conclusions</h2>
                <p>This report presents a comprehensive analysis of different prompting techniques on the AILuminate benchmark.</p>
                <p>Based on the safety scores, {best_technique} shows the best performance in terms of safely handling sensitive prompts.</p>
                <p>The analysis also shows that response times and lengths vary significantly between techniques, with {slowest_technique} taking the longest time on average.</p>
                <p>See the sample-by-sample report for detailed responses for each prompt and technique.</p>
            </div>
        </body>
        </html>
        """.format(
            timestamp=self.results["metadata"]["timestamp"],
            model=self.results["metadata"]["model"],
            techniques=", ".join(self.techniques),
            num_samples=len(self.samples),
            summary_rows="".join([
                f"<tr><td>{row['technique']}</td><td>{row['avg_response_time_sec']:.2f}</td>"
                f"<td>{row['std_response_time_sec']:.2f}</td><td>{row['avg_response_length_chars']:.2f}</td>"
                f"<td>{row['std_response_length_chars']:.2f}</td><td>{row['num_samples']}</td></tr>"
                for _, row in summary_stats.iterrows()
            ]),
            hazard_rows="".join([
                f"<tr><td>{row['hazard']}</td><td>{row['count']}</td></tr>"
                for _, row in hazard_stats.iterrows()
            ]),
            safety_rows="".join([
                f"""<tr class="safety-{'high' if row['safety_score'] >= 70 else 'medium' if row['safety_score'] >= 50 else 'low'}">
                <td>{row['technique']}</td><td>{row['safety_score']:.1f}</td></tr>"""
                for _, row in overall_safety.sort_values("safety_score", ascending=False).iterrows()
            ]),
            best_technique=overall_safety.sort_values("safety_score", ascending=False).iloc[0]["technique"]
                           if not overall_safety.empty else "N/A",
            slowest_technique=summary_stats.sort_values("avg_response_time_sec", ascending=False).iloc[0]["technique"]
                              if not summary_stats.empty else "N/A"
        )
        
        with open(os.path.join(output_dir, "analysis_report.html"), 'w') as f:
            f.write(html)
        
        # Generate sample-by-sample report
        self.generate_sample_report(os.path.join(output_dir, "sample_report.html"))
        
        print(f"Analysis report generated in: {output_dir}")


def main():
    """Main function to run the analyzer from command line."""
    parser = argparse.ArgumentParser(description="Analyze AILuminate benchmark results")
    
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the JSON results file from the benchmark")
    parser.add_argument("--output_dir", type=str, default="analysis",
                        help="Directory to save the analysis reports and plots")
    parser.add_argument("--safety_report", action="store_true",
                        help="Generate a detailed safety report in addition to standard analysis")
    
    args = parser.parse_args()
    
    analyzer = AILuminateAnalyzer(args.results_file)
    analyzer.generate_analysis_report(args.output_dir)
    
    # Generate additional safety report if requested
    if args.safety_report:
        safety_dir = os.path.join(args.output_dir, "safety_analysis")
        safety_scorer = SafetyScorer()
        safety_report = safety_scorer.generate_safety_report(args.results_file, safety_dir)
        print(f"Detailed safety report generated: {safety_report}")


if __name__ == "__main__":
    main() 