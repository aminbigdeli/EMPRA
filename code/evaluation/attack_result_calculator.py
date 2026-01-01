#!/usr/bin/env python3
"""
Calculate attack performance metrics including rank promotion, perplexity, and readability.
This script computes metrics for adversarial documents compared to original target documents.
"""

import argparse
import bisect
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2PPLScorer(object):
    def __init__(self, device):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def perplexity(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = self.tokenizer.batch_encode_plus(inputs, padding='longest', return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()
        shift_masks = inputs['attention_mask'][:, 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape[0], -1) * shift_masks
        loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
        ppl = torch.exp(loss).detach().cpu().numpy().tolist()
        return ppl


class RankPromotionCalculator:
    """Calculate rank promotion metrics for adversarial attacks."""
    
    def __init__(self, rank_list_len: int = 1000):
        if rank_list_len not in [1000, 100]:
            raise ValueError(f"rank_list_len must be 1000 or 100, got {rank_list_len}")
        self.rank_list_len = rank_list_len
    
    def load_attack_scores(self, attack_score_file: Path) -> Dict[str, Dict[str, float]]:
        """Load attack scores from file: qid \t did \t score"""
        attack_scores = defaultdict(dict)
        with open(attack_score_file, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], parts[2]
                    attack_scores[qid][did] = float(score)
        return attack_scores
    
    def load_targets(self, target_file: Path, attack_scores: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, Tuple[int, float]]], int]:
        """Load target documents: qid did rank score"""
        targets = defaultdict(dict)
        target_num = 0
        with open(target_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, did, rank, score = parts[0], parts[1], parts[2], parts[3]
                    if qid in attack_scores and did in attack_scores[qid]:
                        targets[qid][did] = (int(rank), float(score))
                        target_num += 1
        return targets, target_num
    
    def load_rank_scores(self, rank_score_file: Path, targets: Dict[str, Dict[str, Tuple[int, float]]]) -> Dict[str, List[float]]:
        """Load rank scores: qid Q0 did rank score model"""
        rank_scores = defaultdict(lambda: [-1e6] * self.rank_list_len)
        with open(rank_score_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, did, rank, score, _ = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    if qid in targets:
                        rank_idx = self.rank_list_len - int(rank)
                        if 0 <= rank_idx < self.rank_list_len:
                            rank_scores[qid][rank_idx] = float(score)
        return rank_scores
    
    def calculate_metrics(
        self,
        attack_scores: Dict[str, Dict[str, float]],
        targets: Dict[str, Dict[str, Tuple[int, float]]],
        rank_scores: Dict[str, List[float]],
        target_tag: str
    ) -> Dict:
        """Calculate rank promotion metrics."""
        metrics = {
            'rank5': 0,
            'rank10': 0,
            'rank20': 0,
            'rank50': 0,
            'rank100': 0,
            'success_num': 0,
            'same_num': 0,
            'failure_num': 0,
            'avg_boost': 0.0,
            'total_target_docs': 0
        }
        
        target_num = sum(len(docs) for docs in targets.values())
        metrics['total_target_docs'] = target_num
        
        for q_id in targets:
            old_scores = rank_scores[q_id]
            for d_id in targets[q_id]:
                old_rank, old_score = targets[q_id][d_id]
                new_score = attack_scores[q_id][d_id]
                new_rank = self.rank_list_len + 1 - bisect.bisect_right(old_scores, new_score)
                
                boost = old_rank - new_rank
                metrics['avg_boost'] += boost
                
                if new_rank < old_rank:
                    metrics['success_num'] += 1
                elif new_rank == old_rank:
                    metrics['same_num'] += 1
                else:
                    metrics['failure_num'] += 1
                
                if new_rank <= 5:
                    metrics['rank5'] += 1
                if new_rank <= 10:
                    metrics['rank10'] += 1
                if new_rank <= 20:
                    metrics['rank20'] += 1
                if new_rank <= 50:
                    metrics['rank50'] += 1
                if new_rank <= 100:
                    metrics['rank100'] += 1
        
        if target_num > 0:
            metrics['success_rate'] = metrics['success_num'] / target_num
            metrics['same_rate'] = metrics['same_num'] / target_num
            metrics['failure_rate'] = metrics['failure_num'] / target_num
            metrics['rank_promote_top5'] = metrics['rank5'] / target_num
            metrics['rank_promote_top10'] = metrics['rank10'] / target_num
            metrics['rank_promote_top20'] = metrics['rank20'] / target_num
            metrics['rank_promote_top50'] = metrics['rank50'] / target_num
            metrics['rank_promote_top100'] = metrics['rank100'] / target_num
            metrics['average_boost'] = metrics['avg_boost'] / target_num
        else:
            metrics['success_rate'] = 0.0
            metrics['same_rate'] = 0.0
            metrics['failure_rate'] = 0.0
            metrics['rank_promote_top5'] = 0.0
            metrics['rank_promote_top10'] = 0.0
            metrics['rank_promote_top20'] = 0.0
            metrics['rank_promote_top50'] = 0.0
            metrics['rank_promote_top100'] = 0.0
            metrics['average_boost'] = 0.0
        
        return metrics


class PerplexityCalculator:
    """Calculate perplexity scores using GPT-2."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.scorer = None
    
    def _get_scorer(self) -> GPT2PPLScorer:
        """Lazy load the GPT-2 scorer."""
        if self.scorer is None:
            self.scorer = GPT2PPLScorer(self.device)
        return self.scorer
    
    def compute_perplexity(self, text_file: Path, output_ppl_file: Path) -> List[float]:
        """Compute perplexity scores for documents in text file."""
        if output_ppl_file.exists():
            return self._load_existing_perplexity(output_ppl_file)
        
        ppl_scorer = self._get_scorer()
        texts = self._load_texts(text_file)
        
        total_ppl = []
        with open(output_ppl_file, 'w') as w:
            for q_id in tqdm(sorted(texts.keys()), desc="Computing perplexity"):
                input_ids = []
                input_texts = []
                
                for p_id, p_text in texts[q_id]:
                    try:
                        if len(p_text) > 0:
                            input_ids.append(p_id)
                            input_texts.append(str(p_text))
                    except Exception as e:
                        print(f"Error processing text for {p_id}: {e}")
                
                if input_texts:
                    ppl_list = ppl_scorer.perplexity(inputs=input_texts)
                    total_ppl.extend(ppl_list)
                    
                    for i, p_id in enumerate(input_ids):
                        w.write(f'{q_id}\t{p_id}\t{ppl_list[i]}\n')
        
        return total_ppl
    
    def _load_texts(self, text_file: Path) -> Dict[str, List[Tuple[str, str]]]:
        """Load texts from TSV file: qid \t pid \t q_text \t p_text"""
        texts = defaultdict(list)
        df = pd.read_csv(text_file, sep='\t', names=['qid', 'pid', 'q_text', 'p_text'], dtype=str)
        
        for _, row in df.iterrows():
            q_id = str(row['qid'])
            p_id = str(row['pid'])
            p_text = str(row['p_text']) if pd.notna(row['p_text']) else ''
            texts[q_id].append((p_id, p_text))
        
        return texts
    
    def _load_existing_perplexity(self, output_ppl_file: Path) -> List[float]:
        """Load existing perplexity scores from file."""
        total_ppl = []
        with open(output_ppl_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    total_ppl.append(float(parts[2]))
        return total_ppl


def calculate_readability(text: str) -> float:
    """
    Calculate readability score using simple heuristics.
    This is a placeholder - can be replaced with more sophisticated metrics.
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    words = text.split()
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    readability = 100.0 - (avg_sentence_length * 1.5 + avg_word_length * 0.5)
    return max(0.0, min(100.0, readability))


def compute_readability_scores(adv_doc_file: Path) -> List[float]:
    """Compute readability scores for all documents."""
    readability_scores = []
    df = pd.read_csv(adv_doc_file, sep='\t', names=['qid', 'pid', 'q_text', 'p_text'], dtype=str)
    
    for _, row in df.iterrows():
        p_text = str(row['p_text']) if pd.notna(row['p_text']) else ''
        score = calculate_readability(p_text)
        readability_scores.append(score)
    
    return readability_scores


def main():
    parser = argparse.ArgumentParser(
        description='Calculate attack performance metrics including rank promotion, perplexity, and readability'
    )
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device ID (e.g., "0" for GPU 0, or "" for CPU)')
    parser.add_argument('--target_tag', type=str, default='easy-5targets',
                        help='Target tag identifier')
    parser.add_argument('--target_file', type=str, required=True,
                        help='Path to target file')
    parser.add_argument('--adv_doc_file', type=str, default=None,
                        help='Path to adversarial documents file')
    parser.add_argument('--adv_doc_score_file', type=str, required=True,
                        help='Path to adversarial document scores file')
    parser.add_argument('--rank_score_file', type=str, required=True,
                        help='Path to rank score file')
    parser.add_argument('--rank_list_len', type=int, default=1000,
                        choices=[100, 1000],
                        help='Length of rank list')
    parser.add_argument('--output_json_file', type=str, required=True,
                        help='Path to output JSON file for metrics')
    
    args = parser.parse_args()
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device else "cpu")
    
    results = {}
    
    if args.adv_doc_score_file:
        print("Calculating rank promotion metrics...")
        calculator = RankPromotionCalculator(rank_list_len=args.rank_list_len)
        
        attack_scores = calculator.load_attack_scores(Path(args.adv_doc_score_file))
        targets, target_num = calculator.load_targets(Path(args.target_file), attack_scores)
        rank_scores = calculator.load_rank_scores(Path(args.rank_score_file), targets)
        
        metrics = calculator.calculate_metrics(attack_scores, targets, rank_scores, args.target_tag)
        results.update(metrics)
        
        print(f"Rank Promotion Metrics:")
        print(f"  Target docs: {metrics['total_target_docs']}")
        print(f"  Success rate: {metrics['success_rate']:.4f}")
        print(f"  Top-10 ratio: {metrics['rank_promote_top10']:.4f}")
        print(f"  Top-50 ratio: {metrics['rank_promote_top50']:.4f}")
        print(f"  Average boost: {metrics['average_boost']:.2f}")
    
    if args.adv_doc_file:
        print("\nComputing perplexity scores...")
        ppl_calculator = PerplexityCalculator(device)
        output_ppl_file = Path(args.adv_doc_file).with_suffix('.ppl')
        ppl_scores = ppl_calculator.compute_perplexity(Path(args.adv_doc_file), output_ppl_file)
        
        if ppl_scores:
            avg_ppl = sum(ppl_scores) / len(ppl_scores)
            results['average_perplexity'] = avg_ppl
            print(f"  Average perplexity: {avg_ppl:.2f}")
        
        print("\nComputing readability scores...")
        readability_scores = compute_readability_scores(Path(args.adv_doc_file))
        if readability_scores:
            avg_readability = sum(readability_scores) / len(readability_scores)
            results['average_readability'] = avg_readability
            print(f"  Average readability: {avg_readability:.2f}")
    
    aggregate_metrics = {
        'success_rate': results.get('success_rate', 0.0),
        'rank_promote_top10': results.get('rank_promote_top10', 0.0),
        'rank_promote_top50': results.get('rank_promote_top50', 0.0),
        'average_boost': results.get('average_boost', 0.0),
        'average_perplexity': results.get('average_perplexity'),
        'average_readability': results.get('average_readability', 0.0)
    }
    
    output_data = {
        'total_target_docs': results.get('total_target_docs', 0),
        'aggregate_metrics': aggregate_metrics,
        'detailed_metrics': results
    }
    
    with open(args.output_json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nMetrics saved to: {args.output_json_file}")


if __name__ == "__main__":
    main()

