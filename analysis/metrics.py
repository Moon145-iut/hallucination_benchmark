import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_fine_grained_metrics(results):
    """
    Analyze results using fine-grained hallucination metrics
    """
    scores = [r['hallucination_metrics']['score'] for r in results]
    confidences = [r['hallucination_metrics']['confidence'] for r in results]
    severities = [r['hallucination_metrics']['severity'] for r in results]
    correctness = [1 if r.get('judgement') == r.get('ground_truth') else 0 for r in results]
    
    # Basic statistics
    stats = {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_confidence': np.mean(confidences),
        'severity_distribution': {
            'factual': severities.count('factual'),
            'slightly_incorrect': severities.count('slightly_incorrect'),
            'strongly_hallucinated': severities.count('strongly_hallucinated')
        },
        'accuracy_severity_correlation': float(np.nan_to_num(np.corrcoef(scores, correctness)[0,1])) if len(set(scores)) > 1 and len(set(correctness)) > 1 else 0.0
    }
    
    # Component analysis
    component_scores = {
        'llm_scores': [r['hallucination_metrics']['components']['llm_score'] for r in results],
        'algorithmic_scores': [r['hallucination_metrics']['components']['algorithmic_score'] for r in results],
        'entity_mismatch': [r['hallucination_metrics']['components']['entity_mismatch'] for r in results],
        'fact_contradiction': [r['hallucination_metrics']['components']['fact_contradiction'] for r in results],
        'semantic_drift': [r['hallucination_metrics']['components']['semantic_drift'] for r in results]
    }
    
    # Calculate correlations
    correlations = {}
    for comp1 in component_scores:
        correlations[comp1] = {}
        for comp2 in component_scores:
            if comp1 != comp2:
                corr = np.corrcoef(component_scores[comp1], component_scores[comp2])[0,1]
                correlations[comp1][comp2] = corr
    
    return {
        'basic_stats': stats,
        'component_scores': component_scores,
        'correlations': correlations
    }

def visualize_results(analysis_results, output_dir):
    """
    Generate visualizations for hallucination analysis
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Severity Distribution
    plt.figure(figsize=(10, 6))
    dist = analysis_results['basic_stats']['severity_distribution']
    plt.bar(dist.keys(), dist.values())
    plt.title('Distribution of Hallucination Severity')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/severity_distribution.png')
    plt.close()
    
    # 2. Component Score Distributions
    plt.figure(figsize=(12, 6))
    comp_scores = analysis_results['component_scores']
    plt.boxplot([comp_scores[k] for k in comp_scores.keys()], labels=comp_scores.keys())
    plt.title('Distribution of Component Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_distributions.png')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = np.array([[analysis_results['correlations'][c1][c2] 
                            if c1 != c2 else 1.0 
                            for c2 in comp_scores.keys()] 
                           for c1 in comp_scores.keys()])
    sns.heatmap(corr_matrix, 
                xticklabels=comp_scores.keys(),
                yticklabels=comp_scores.keys(),
                annot=True,
                cmap='coolwarm')
    plt.title('Component Score Correlations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlations.png')
    plt.close()

def generate_report(analysis_results, output_file):
    """
    Generate a detailed analysis report
    """
    report = {
        'summary_statistics': {
            'average_hallucination_score': analysis_results['basic_stats']['avg_score'],
            'score_standard_deviation': analysis_results['basic_stats']['std_score'],
            'average_confidence': analysis_results['basic_stats']['avg_confidence']
        },
        'severity_distribution': analysis_results['basic_stats']['severity_distribution'],
        'component_analysis': {
            'average_scores': {
                comp: np.mean(scores)
                for comp, scores in analysis_results['component_scores'].items()
            }
        },
        'correlations': analysis_results['correlations']
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
