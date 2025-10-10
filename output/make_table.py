import json
import os
import pandas as pd
from pathlib import Path
from utils import get_score_from_file

def parse_filename(filename):
    """Parse filename to extract model, dataset, split, and method."""
    filename = filename.replace('.json', '')
    
    model_prefixes = [
        "DeepSeek-R1", "DeepSeek-V3", "Llama-3.3-70B-Instruct-Turbo", 
        "QwQ-32B-Preview", "QwQ-32B", "claude-3-5-haiku", "claude-3-5-sonnet",
        "gpt-35-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"
    ]
    
    model = None
    for prefix in sorted(model_prefixes, key=len, reverse=True):
        if filename.startswith(prefix):
            model = prefix
            remaining = filename[len(prefix):].lstrip('-')
            break
    
    if not model:
        return None, None, None, None
    
    datasets = [
        "medqa", "medbullets", "mmlu", "pubmedqa", "medexqa", "medmcqa", 
        "medxpertqa-r", "medxpertqa-u", "mmlu-pro"
    ]
    
    dataset = None
    for ds in sorted(datasets, key=len, reverse=True):
        if ds in remaining:
            dataset = ds
            remaining = remaining.replace(ds, '', 1).strip('-')
            break
    
    if not dataset:
        return None, None, None, None
    
    splits = ["test_hard", "test"]
    split = None
    for s in splits:
        if s in remaining:
            split = s
            remaining = remaining.replace(s, '', 1).strip('-')
            break
    
    if not split:
        return None, None, None, None
    
    method = remaining if remaining else "zero_shot"
    
    return model, dataset, split, method

def parse_method(method):
    """Parse method name to return standardized display name."""
    if method == "zero_shot":
        return "Zero-shot"
    elif method == "cot_sc-5":
        return "CoT-SC"
    elif method == "aflow":
        return "AFlow"
    elif method == "mdagents":
        return "MDAgents"
    elif method == "few_shot":
        return "Few-shot"
    elif method == "medagents":
        return "MedAgents"
    elif method == "self_refine-3":
        return "Self-refine"
    elif method == "medprompt-3":
        return "MedPrompt"
    elif method == "multipersona-2":
        return "MultiPersona"
    elif method == "cot":
        return "CoT"
    elif method == "spo":
        return "SPO"
    else:
        return method

def create_results_table():
    """Create a comprehensive results table from all JSON files."""
    results_data = []
    
    datasets = ["medqa", "medbullets", "mmlu", "pubmedqa", "medexqa", "medmcqa", "medxpertqa-r", "medxpertqa-u", "mmlu-pro"]
    model_names = ["o3-mini", "gpt-4o", "gpt-4o-mini"]
    split = "test_hard"
    run_ids = [0, 1, 2]
    
    output_dir = Path('.')
    
    for run_id in run_ids:
        run_dir = output_dir / f'run-{run_id}'
        if not run_dir.is_dir():
            continue

        for dataset in datasets:
            dataset_dir = run_dir / dataset
            if not dataset_dir.is_dir():
                continue
            
            for json_file in dataset_dir.iterdir():
                if not json_file.is_file() or not json_file.name.endswith('.json'):
                    continue
                
                parsed_model, parsed_dataset, parsed_split, method = parse_filename(json_file.name)
                
                if parsed_model not in model_names or parsed_dataset != dataset or parsed_split != split:
                    continue
                
                accuracy, count, cost, time = get_score_from_file(json_file, parsed_model)
                
                if accuracy is not None:
                    model_csv_name = parsed_model.replace("-", "_")
                    results_data.append({
                        'exp_name': parse_method(method),
                        'dataset': dataset,
                        'model': model_csv_name,
                        'run_id': run_id,
                        'accuracy': round(accuracy, 1),
                        'avg_time': round(time, 2),
                        'avg_cost': round(cost, 4),
                        'count': count
                    })
    
    df = pd.DataFrame(results_data)
    
    if df.empty:
        print("No results found!")
        return df
    
    df = df.sort_values(['dataset', 'model', 'run_id'])
    
    return df

def main():
    """Main function to generate and display results tables."""
    print("Loading results from all JSON files...")
    df = create_results_table()
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"\nFound {len(df)} results across {df['dataset'].nunique()} datasets and {df['model'].nunique()} models")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE - All Results")
    print("="*80)
    print(df.to_string(index=False))
    
    csv_file = 'baseline_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

if __name__ == "__main__":
    main()
