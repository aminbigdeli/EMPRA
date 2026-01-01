#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class COLAScorer:
    
    def __init__(self, device: torch.device = None, model_name: str = "textattack/roberta-base-CoLA"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading CoLA model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def predict_acceptability(self, document: str) -> float:
        if not document or len(document.strip()) == 0:
            return 0.0
        
        inputs = self.tokenizer(document, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            acceptability_score = torch.softmax(logits, dim=-1)[0][1].item()
        
        return acceptability_score
    
    def predict_batch(self, documents: List[str], batch_size: int = 32) -> List[float]:
        if not documents:
            return []
        
        scores = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            
            scores.extend(batch_scores)
        
        return scores


def load_documents(input_file: Path, document_column: str = "document") -> pd.DataFrame:
    df = pd.read_csv(input_file, sep='\t', dtype=str)
    
    if document_column not in df.columns:
        raise ValueError(f"Column '{document_column}' not found in input file. Available columns: {list(df.columns)}")
    
    return df


def calculate_acceptability_scores(
    df: pd.DataFrame,
    scorer: COLAScorer,
    document_column: str = "document",
    batch_size: int = 32
) -> pd.DataFrame:
    documents = df[document_column].fillna('').astype(str).tolist()
    
    print(f"Calculating acceptability scores for {len(documents)} documents...")
    scores = scorer.predict_batch(documents, batch_size=batch_size)
    
    df['acceptability_score'] = scores
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate linguistic acceptability scores using CoLA model'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input TSV file containing documents'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to output TSV file (default: input_file with _cola suffix)'
    )
    parser.add_argument(
        '--document_column',
        type=str,
        default='document',
        help='Name of column containing documents (default: document)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='textattack/roberta-base-CoLA',
        help='HuggingFace model name for CoLA (default: textattack/roberta-base-CoLA)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing documents (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run model on (default: auto)'
    )
    parser.add_argument(
        '--gpu_id',
        type=str,
        default=None,
        help='CUDA device ID (e.g., "0" for GPU 0). Sets CUDA_VISIBLE_DEVICES if provided.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Acceptability threshold for counting acceptable documents (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.parent / f"{input_file.stem}_cola{input_file.suffix}"
    
    print(f"Loading documents from {input_file}")
    df = load_documents(input_file, document_column=args.document_column)
    print(f"Loaded {len(df)} documents")
    
    scorer = COLAScorer(device=device, model_name=args.model_name)
    
    df = calculate_acceptability_scores(
        df,
        scorer,
        document_column=args.document_column,
        batch_size=args.batch_size
    )
    
    num_acceptable = (df['acceptability_score'] >= args.threshold).sum()
    num_total = len(df)
    avg_score = df['acceptability_score'].mean()
    min_score = df['acceptability_score'].min()
    max_score = df['acceptability_score'].max()
    
    print(f"\n{'='*60}")
    print("Linguistic Acceptability Statistics")
    print(f"{'='*60}")
    print(f"Total documents: {num_total}")
    print(f"Acceptable documents (score >= {args.threshold}): {num_acceptable} ({100*num_acceptable/num_total:.2f}%)")
    print(f"Average acceptability score: {avg_score:.4f}")
    print(f"Min score: {min_score:.4f}")
    print(f"Max score: {max_score:.4f}")
    print(f"{'='*60}")
    
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
