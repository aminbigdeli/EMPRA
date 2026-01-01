#!/usr/bin/env python3
"""
End-to-end pipeline to process adversarial perturbation TSV files:
1. Score query-document pairs using CrossEncoder
2. Generate adv_doc_file and adv_doc_score_file
3. Run attack_result_calculator.py for each method
4. Organize outputs by method
5. Create summary TSV with aggregate metrics
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


def sanitize_text(text: str) -> str:
    """Sanitize text for TSV format."""
    if text is None:
        return ''
    return str(text).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')


def extract_base_doc_id(doc_id: str, method: str) -> str:
    """
    Extract base doc_id from compound doc_ids.
    For EMPRA and IDEM methods, doc_ids are like "1587491-1-0",
    extract the first number before the first '-' as the base doc_id.
    For other methods, return doc_id as-is.
    """
    if method.upper() in ['EMPRA', 'IDEM']:
        # Extract first number before first '-'
        if '-' in doc_id:
            base_id = doc_id.split('-')[0]
            return base_id
    return doc_id


def score_query_document_pairs(
    input_tsv: Path,
    model: CrossEncoder,
    batch_size: int = 64,
    method: str = None
) -> Tuple[List[Tuple[str, str, str, str]], List[float]]:
    """
    Read TSV file and score all query-document pairs.
    Returns: (pairs_data, scores)
    pairs_data: list of (qid, doc_id, query, document)
    """
    # Check if file is BrittleBERT - read manually instead of using pandas
    is_brittlebert = 'BrittleBERT' in input_tsv.name or 'BrittleBert' in input_tsv.name
    
    pairs_data = []
    query_doc_pairs = []
    
    if is_brittlebert:
        # Read file manually for BrittleBERT files
        with open(input_tsv, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    qid = str(parts[0])
                    doc_id = str(parts[1])
                    # Extract base doc_id for EMPRA and IDEM methods
                    if method:
                        doc_id = extract_base_doc_id(doc_id, method)
                    query = sanitize_text(parts[2])
                    document = sanitize_text(parts[3])
                    
                    pairs_data.append((qid, doc_id, query, document))
                    query_doc_pairs.append((query, document))
    else:
        # Use pandas for other files
        df = pd.read_csv(input_tsv, sep='\t', names=['qid', 'doc_id', 'query', 'document'], dtype=str)
        
        for _, row in df.iterrows():
            qid = str(row['qid'])
            doc_id = str(row['doc_id'])
            # Extract base doc_id for EMPRA and IDEM methods
            if method:
                doc_id = extract_base_doc_id(doc_id, method)
            query = sanitize_text(row['query'])
            document = sanitize_text(row['document'])
            
            pairs_data.append((qid, doc_id, query, document))
            query_doc_pairs.append((query, document))
    
    # Score in batches
    scores = []
    total_batches = (len(query_doc_pairs) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(query_doc_pairs), batch_size), 
                  desc=f"Scoring {input_tsv.name}", total=total_batches):
        batch = query_doc_pairs[i:i+batch_size]
        batch_scores = model.predict(batch, convert_to_tensor=False, show_progress_bar=False)
        scores.extend([float(s) for s in batch_scores])
    
    return pairs_data, scores


def create_adv_files(
    pairs_data: List[Tuple[str, str, str, str]],
    scores: List[float],
    output_dir: Path,
    method_name: str,
    tag: str
) -> Tuple[Path, Path]:
    """
    Create adv_doc_file and adv_doc_score_file.
    Returns: (adv_doc_file_path, adv_doc_score_file_path)
    """
    adv_doc_file = output_dir / f"{tag}-{method_name}_adv_docs.tsv"
    adv_doc_score_file = output_dir / f"{tag}-{method_name}_adv_scores.tsv"
    
    # Write adv_doc_file: qid \t pid \t q_text \t p_text
    with open(adv_doc_file, 'w', encoding='utf-8') as f:
        for qid, doc_id, query, document in pairs_data:
            f.write(f"{qid}\t{doc_id}\t{query}\t{document}\n")
    
    # Write adv_doc_score_file: qid \t did \t score
    with open(adv_doc_score_file, 'w', encoding='utf-8') as f:
        for (qid, doc_id, _, _), score in zip(pairs_data, scores):
            f.write(f"{qid}\t{doc_id}\t{score}\n")
    
    return adv_doc_file, adv_doc_score_file


def run_attack_calculator(
    adv_doc_file: Path,
    adv_doc_score_file: Path,
    target_file: Path,
    rank_score_file: Path,
    output_json_file: Path,
    target_tag: str,
    rank_list_len: int,
    gpu_id: str
) -> bool:
    """Run attack_result_calculator.py script."""
    script_path = Path(__file__).parent.parent / "evaluation" / "attack_result_calculator.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        '--device', gpu_id,
        '--target_tag', target_tag,
        '--target_file', str(target_file),
        '--adv_doc_file', str(adv_doc_file),
        '--adv_doc_score_file', str(adv_doc_score_file),
        '--rank_score_file', str(rank_score_file),
        '--rank_list_len', str(rank_list_len),
        '--output_json_file', str(output_json_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running attack_result_calculator: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def extract_aggregate_metrics(json_file: Path) -> Dict:
    """Extract aggregate metrics from JSON output."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('aggregate_metrics', {})
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return {}


def create_summary_tsv(
    results: Dict[str, Dict[str, Dict]],
    output_file: Path,
    append: bool = False
):
    """Create summary TSV file with average metrics per method and tag."""
    rows = []
    
    for tag in sorted(results.keys()):
        for method in sorted(results[tag].keys()):
            metrics = results[tag][method]
            if not metrics:
                continue
            
            # Extract metrics
            n = metrics.get('total_target_docs', 0)
            success_rate = metrics.get('success_rate', 0) * 100
            top10_ratio = metrics.get('rank_promote_top10', 0) * 100
            top50_ratio = metrics.get('rank_promote_top50', 0) * 100
            avg_boost = metrics.get('average_boost', 0)
            avg_perplexity = metrics.get('average_perplexity')
            avg_readability = metrics.get('average_readability', 0)
            
            rows.append({
                'tag': tag.capitalize(),
                'method': method,
                'n': n,
                'attack_success_rate': round(success_rate, 1),
                'top10_ratio': round(top10_ratio, 1),
                'top50_ratio': round(top50_ratio, 1),
                'avg_boost': round(avg_boost, 1),
                'avg_perplexity': round(avg_perplexity, 1) if avg_perplexity is not None else 0,
                'avg_readability': round(avg_readability, 1)
            })
    
    # Read existing data if appending
    if append and output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, sep='\t')
            # Combine with new rows
            new_df = pd.DataFrame(rows)
            # Remove duplicates based on tag and method
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates, keeping the last occurrence (new data)
            combined_df = combined_df.drop_duplicates(subset=['tag', 'method'], keep='last')
            df = combined_df.sort_values(by=['tag', 'method'])
        except Exception as e:
            print(f"Warning: Could not read existing summary file: {e}")
            print("Creating new summary file instead.")
            df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows)
    
    # Write TSV
    df.to_csv(output_file, sep='\t', index=False)
    mode = "appended to" if append and output_file.exists() else "saved to"
    print(f"Summary TSV {mode} {output_file}")


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract tag and method.
    Format: {tag}-{method}.tsv
    """
    base = filename.replace('.tsv', '')
    parts = base.split('-', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return 'unknown', base


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline for processing adversarial perturbations'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., trecdl2019, trecdl2020)'
    )
    parser.add_argument(
        '--tag',
        type=str,
        required=True,
        choices=['S1', 'S3'],
        help='Tag (S1 or S3)'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Input directory containing TSV files (required if --input_file not provided)'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Specific TSV file to process (if provided, only this file will be processed)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/abigdeli/projects/adv_ir/EMPRA/data',
        help='Base data directory containing target and rank score files'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='/home/abigdeli/projects/adv_ir/EMPRA/ablation_experiments/dataset_generalization/results',
        help='Base output directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cross-encoder/ms-marco-MiniLM-L-12-v2',
        help='CrossEncoder model name'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for CrossEncoder scoring'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run CrossEncoder on (auto/cpu/cuda)'
    )
    parser.add_argument(
        '--gpu_id',
        type=str,
        default='0',
        help='GPU device ID for attack_result_calculator (e.g., "0" for GPU 0, or "" for CPU)'
    )
    parser.add_argument(
        '--rank_list_len',
        type=int,
        default=1000,
        help='Length of rank list (1000 or 100)'
    )
    parser.add_argument(
        '--target_file_easy',
        type=str,
        default=None,
        help='Path to easy target file (if not provided, will be constructed from dataset and data_dir)'
    )
    parser.add_argument(
        '--target_file_hard',
        type=str,
        default=None,
        help='Path to hard target file (if not provided, will be constructed from dataset and data_dir)'
    )
    parser.add_argument(
        '--rank_score_file',
        type=str,
        default=None,
        help='Path to rank score file (if not provided, will be constructed from dataset, data_dir, and tag)'
    )
    
    args = parser.parse_args()
    
    # Setup device for CrossEncoder
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    
    # Validate input arguments
    if not args.input_file and not args.input_dir:
        print("Error: Either --input_file or --input_dir must be provided")
        return
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_base_dir = Path(args.output_base_dir)
    
    # Use user-provided paths or construct from dataset
    if args.target_file_easy:
        target_file_easy = Path(args.target_file_easy)
    else:
        dataset_dir = data_dir / args.dataset
        target_file_easy = dataset_dir / f"rerank.easy-5targets.tsv"
    
    if args.target_file_hard:
        target_file_hard = Path(args.target_file_hard)
    else:
        dataset_dir = data_dir / args.dataset
        target_file_hard = dataset_dir / f"rerank.hard-5targets.tsv"
    
    if args.rank_score_file:
        rank_score_file = Path(args.rank_score_file)
    else:
        dataset_dir = data_dir / args.dataset
        rank_score_file = dataset_dir / f"reranked_run_{args.tag}.trec"
    
    # Check if files exist
    if not target_file_easy.exists():
        print(f"Error: {target_file_easy} not found")
        return
    if not target_file_hard.exists():
        print(f"Error: {target_file_hard} not found")
        return
    if not rank_score_file.exists():
        print(f"Error: {rank_score_file} not found")
        return
    
    # Output directory structure: {output_base_dir}/{dataset}/{tag}/
    output_dir = output_base_dir / args.dataset / f"adv_perturbations_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CrossEncoder model
    print(f"Loading CrossEncoder model: {args.model} on {device_str}")
    model = CrossEncoder(args.model, max_length=512, device=device_str)
    
    # Find TSV files to process
    if args.input_file:
        # Process single file
        input_file_path = Path(args.input_file)
        if not input_file_path.exists():
            print(f"Error: Input file {input_file_path} not found")
            return
        if not input_file_path.suffix == '.tsv':
            print(f"Error: Input file must be a .tsv file")
            return
        tsv_files = [input_file_path]
        print(f"Processing single file: {input_file_path.name}")
    else:
        # Process all files in directory
        input_dir = Path(args.input_dir)
        tsv_files = sorted(input_dir.glob("*.tsv"))
        if not tsv_files:
            print(f"No TSV files found in {input_dir}")
            return
        print(f"Found {len(tsv_files)} TSV files to process")
    
    # Store results for summary
    results = defaultdict(lambda: defaultdict(dict))
    
    # Process each TSV file
    for tsv_file in tsv_files:
        print(f"\n{'='*80}")
        print(f"Processing: {tsv_file.name}")
        print(f"{'='*80}")
        
        # Parse filename to get tag and method
        tag, method = parse_filename(tsv_file.name)
        tag_lower = tag.lower()
        
        # Select appropriate target file
        if tag_lower == 'easy':
            target_file = target_file_easy
        elif tag_lower == 'hard':
            target_file = target_file_hard
        else:
            print(f"Warning: Unknown tag '{tag}', skipping {tsv_file.name}")
            continue
        
        # Create method-specific output directory
        method_output_dir = output_dir / method
        method_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Score query-document pairs
        print("Step 1: Scoring query-document pairs...")
        pairs_data, scores = score_query_document_pairs(
            tsv_file, model, args.batch_size, method=method
        )
        print(f"Scored {len(scores)} pairs")
        
        # Step 2: Create adv_doc_file and adv_doc_score_file
        print("Step 2: Creating adv_doc_file and adv_doc_score_file...")
        adv_doc_file, adv_doc_score_file = create_adv_files(
            pairs_data, scores, method_output_dir, method, tag_lower
        )
        print(f"Created: {adv_doc_file.name}")
        print(f"Created: {adv_doc_score_file.name}")
        
        # Step 3: Run attack_result_calculator
        print("Step 3: Running attack_result_calculator...")
        output_json_file = method_output_dir / f"{tag_lower}-{method}_metrics.json"
        target_tag = f"{tag_lower}-5targets"
        
        success = run_attack_calculator(
            adv_doc_file=adv_doc_file,
            adv_doc_score_file=adv_doc_score_file,
            target_file=target_file,
            rank_score_file=rank_score_file,
            output_json_file=output_json_file,
            target_tag=target_tag,
            rank_list_len=args.rank_list_len,
            gpu_id=args.gpu_id if device_str == 'cuda' else ''
        )
        
        if success:
            print(f"Metrics saved to: {output_json_file}")
            # Extract aggregate metrics for summary
            metrics = extract_aggregate_metrics(output_json_file)
            # Also get total_target_docs
            try:
                with open(output_json_file, 'r') as f:
                    data = json.load(f)
                metrics['total_target_docs'] = data.get('total_target_docs', 0)
            except:
                pass
            results[tag_lower][method] = metrics
        else:
            print(f"Failed to process {tsv_file.name}")
    
    # Step 4: Create summary TSV
    print(f"\n{'='*80}")
    print("Creating summary TSV...")
    print(f"{'='*80}")
    summary_file = output_dir / "metrics_summary.tsv"
    append_mode = summary_file.exists()
    if append_mode:
        print(f"Summary file exists. Appending new results...")
    create_summary_tsv(results, summary_file, append=append_mode)
    
    print(f"\n{'='*80}")
    print("Pipeline completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

