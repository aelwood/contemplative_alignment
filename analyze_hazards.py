import json
from collections import defaultdict

# Load the detailed evaluations
with open('large_benchmark_results/anova_analysis/safety_analysis/detailed_evaluations_20250329_193309.json', 'r') as f:
    data = json.load(f)

# Create a nested defaultdict to store scores by hazard and technique
hazard_scores = defaultdict(lambda: defaultdict(list))

# Process each evaluation
for eval in data.get('evaluations', []):
    hazard = eval.get('hazard', 'unknown')
    techniques = eval.get('techniques', {})
    
    # Extract scores for each technique
    for tech_name, tech_data in techniques.items():
        safety_score = tech_data.get('evaluation', {}).get('safety_score', 0)
        hazard_scores[hazard][tech_name].append(safety_score)

# Calculate average scores for each hazard and technique
hazard_averages = {}
for hazard, tech_scores in hazard_scores.items():
    hazard_averages[hazard] = {
        tech: sum(scores)/len(scores) for tech, scores in tech_scores.items()
    }

# Print the results, sorted by hazard category
print('Average Safety Scores by Hazard Category:')
for hazard, tech_avg in sorted(hazard_averages.items()):
    print(f'\n{hazard}:')
    # Sort techniques by average score (descending)
    for tech, avg in sorted(tech_avg.items(), key=lambda x: x[1], reverse=True):
        print(f'  {tech}: {avg:.2f}')

# Calculate the technique rankings across hazards
technique_ranks = defaultdict(list)
for hazard, tech_scores in hazard_averages.items():
    # Sort techniques by score in descending order
    ranked_techniques = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (tech, _) in enumerate(ranked_techniques, 1):
        technique_ranks[tech].append(rank)

# Calculate average rank for each technique
print('\nAverage Technique Rankings Across Hazards:')
avg_ranks = {tech: sum(ranks)/len(ranks) for tech, ranks in technique_ranks.items()}
for tech, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
    print(f'  {tech}: {avg_rank:.2f}')

# Print the number of samples per hazard category
print('\nNumber of Samples per Hazard Category:')
for hazard, tech_scores in hazard_scores.items():
    # Just take the first technique to count samples
    first_tech = next(iter(tech_scores.values()))
    print(f'  {hazard}: {len(first_tech)} samples') 