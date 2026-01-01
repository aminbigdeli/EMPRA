#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ContentFidelityEvaluator:
    
    def __init__(self, device: torch.device = None, nli_model_name: str = "microsoft/deberta-large-mnli"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.nli_model_name = nli_model_name
        
        print(f"Loading NLI model: {nli_model_name} on {device}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
        self.nli_model.eval()
        
        for param in self.nli_model.parameters():
            param.requires_grad = False
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def truncate_text(self, text: str, max_tokens: int = 400) -> str:
        words = str(text).split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens])
        return str(text)
    
    def compute_rouge_l(self, original_texts: List[str], adversarial_texts: List[str]) -> float:
        rouge_l_scores = []
        
        for orig, adv in zip(original_texts, adversarial_texts):
            scores = self.rouge_scorer.score(str(orig), str(adv))
            rouge_l_scores.append(scores['rougeL'].recall)
        
        return np.mean(rouge_l_scores) if rouge_l_scores else 0.0
    
    def compute_backward_entailment(self, original_texts: List[str], adversarial_texts: List[str], batch_size: int = 32) -> float:
        entailment_scores = []
        
        for i in range(0, len(original_texts), batch_size):
            batch_orig = original_texts[i:i + batch_size]
            batch_adv = adversarial_texts[i:i + batch_size]
            
            batch_scores = []
            
            for orig, adv in zip(batch_orig, batch_adv):
                orig_trunc = self.truncate_text(orig)
                adv_trunc = self.truncate_text(adv)
                
                inputs = self.nli_tokenizer(
                    adv_trunc,
                    orig_trunc,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1)[0]
                entailment_prob = probs[2].item()
                batch_scores.append(entailment_prob)
            
            entailment_scores.extend(batch_scores)
        
        return np.mean(entailment_scores) if entailment_scores else 0.0
    
    def compute_bertscore_f1(self, original_texts: List[str], adversarial_texts: List[str], model_type: str = 'roberta-large', batch_size: int = 64) -> float:
        original_texts = [str(t) for t in original_texts]
        adversarial_texts = [str(t) for t in adversarial_texts]
        
        P, R, F1 = bert_score(
            adversarial_texts,
            original_texts,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
            device=self.device
        )
        
        return np.mean(F1.numpy())


def load_collection(collection_file_path: Path) -> Dict[str, str]:
    print(f"Loading collection from {collection_file_path}")
    df = pd.read_csv(collection_file_path, sep='\t', header=None, names=['doc_id', 'doc_text'], dtype=str)
    print(f"Loaded {len(df)} documents from collection")
    doc_dict = dict(zip(df['doc_id'].astype(str), df['doc_text']))
    return doc_dict


def load_adversarial_docs(adv_file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(adv_file_path, sep='\t', header=None, names=['qid', 'doc_id', 'query', 'adv_doc'], dtype=str)
    return df


def extract_base_doc_id(doc_id: str) -> str:
    return str(doc_id).split('-')[0]


def process_file(
    adv_file_path: Path,
    collection_dict: Dict[str, str],
    evaluator: ContentFidelityEvaluator
) -> Dict:
    print(f"\n  Processing: {adv_file_path.name}")
    
    adv_df = load_adversarial_docs(adv_file_path)
    
    original_texts = []
    adversarial_texts = []
    missing_count = 0
    
    for idx, row in adv_df.iterrows():
        doc_id = str(row['doc_id'])
        adv_doc = str(row['adv_doc'])
        
        base_doc_id = extract_base_doc_id(doc_id)
        
        if base_doc_id not in collection_dict:
            missing_count += 1
            continue
        
        original_texts.append(collection_dict[base_doc_id])
        adversarial_texts.append(adv_doc)
    
    if missing_count > 0:
        print(f"    Warning: {missing_count} documents not found in collection")
    
    if not original_texts:
        print(f"    Error: No valid document pairs found!")
        return None
    
    print(f"    Processing {len(original_texts)} document pairs...")
    
    print(f"    Computing ROUGE-L recall...")
    rouge_l = evaluator.compute_rouge_l(original_texts, adversarial_texts)
    
    print(f"    Computing Backward Entailment...")
    backward_entailment = evaluator.compute_backward_entailment(original_texts, adversarial_texts)
    
    print(f"    Computing BERTScore F1...")
    bertscore_f1 = evaluator.compute_bertscore_f1(original_texts, adversarial_texts)
    
    results = {
        'filename': adv_file_path.name,
        'num_docs': len(original_texts),
        'rougeL_recall': rouge_l,
        'backward_entailment': backward_entailment,
        'bertscore_f1': bertscore_f1
    }
    
    return results


def process_folder(
    folder_path: Path,
    collection_dict: Dict[str, str],
    evaluator: ContentFidelityEvaluator,
    output_csv: Path
):
    print(f"\n{'='*70}")
    print(f"Processing folder: {folder_path}")
    print(f"{'='*70}")
    
    tsv_files = sorted(folder_path.glob('*.tsv'))
    
    if not tsv_files:
        print(f"No TSV files found in {folder_path}")
        return
    
    print(f"Found {len(tsv_files)} TSV files")
    
    results = []
    
    for tsv_file in tsv_files:
        try:
            result = process_file(tsv_file, collection_dict, evaluator)
            if result:
                results.append(result)
        except Exception as e:
            print(f"    Error processing {tsv_file}: {e}")
            continue
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_csv}")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))
    else:
        print(f"\nNo results to save for {folder_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate content fidelity metrics (ROUGE-L, Backward Entailment, BERTScore F1) on adversarial documents'
    )
    parser.add_argument(
        '--collection_file',
        type=str,
        required=True,
        help='Path to collection TSV file (doc_id, doc_text)'
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        default=None,
        help='Path to folder containing adversarial TSV files'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Path to single adversarial TSV file to process'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path to output CSV file (default: input_folder/results.csv or input_file_results.csv)'
    )
    parser.add_argument(
        '--nli_model',
        type=str,
        default='microsoft/deberta-large-mnli',
        help='NLI model name (default: microsoft/deberta-large-mnli)'
    )
    parser.add_argument(
        '--bertscore_model',
        type=str,
        default='roberta-large',
        help='BERTScore model type (default: roberta-large)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run models on (default: auto)'
    )
    parser.add_argument(
        '--gpu_id',
        type=str,
        default=None,
        help='CUDA device ID (e.g., "0" for GPU 0). Sets CUDA_VISIBLE_DEVICES if provided.'
    )
    
    args = parser.parse_args()
    
    if not args.input_folder and not args.input_file:
        print("Error: Either --input_folder or --input_file must be provided")
        return
    
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print(f"\nDevice: {device}")
    
    collection_file = Path(args.collection_file)
    if not collection_file.exists():
        print(f"Error: Collection file {collection_file} not found")
        return
    
    collection_dict = load_collection(collection_file)
    
    evaluator = ContentFidelityEvaluator(device=device, nli_model_name=args.nli_model)
    
    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: Input file {input_file} not found")
            return
        
        if args.output_csv:
            output_csv = Path(args.output_csv)
        else:
            output_csv = input_file.parent / f"{input_file.stem}_content_fidelity_results.csv"
        
        result = process_file(input_file, collection_dict, evaluator)
        if result:
            results_df = pd.DataFrame([result])
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
            print(results_df.to_string(index=False))
    else:
        input_folder = Path(args.input_folder)
        if not input_folder.exists():
            print(f"Error: Input folder {input_folder} not found")
            return
        
        if args.output_csv:
            output_csv = Path(args.output_csv)
        else:
            output_csv = input_folder / "content_fidelity_results.csv"
        
        process_folder(input_folder, collection_dict, evaluator, output_csv)
    
    print(f"\n{'='*70}")
    print("Content fidelity evaluation completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

