import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from attack.sentence_merger import SentenceMergerConfig, SentenceMergerPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Construct adversarial documents by merging generated sentences with original documents and selecting best based on coherence and relevance scores"
    )
    
    parser.add_argument('--relevance-model', type=str, required=True,
                        help='Path to BERT model directory for relevance scoring')
    parser.add_argument('--model-tag', type=str, default=None,
                        help='Tag for model (used in output filenames)')
    parser.add_argument('--connect-sent-file', type=str, required=True,
                        help='Path to file containing connection sentences')
    parser.add_argument('--target-file', type=str, required=True,
                        help='Path to target documents file')
    parser.add_argument('--query-collection', type=str, required=True,
                        help='Path to query collection TSV file')
    parser.add_argument('--doc-collection', type=str, required=True,
                        help='Path to document collection TSV file')
    
    parser.add_argument('--coh-weight', type=float, default=0.5,
                        help='Weight for coherence score (default: 0.5)')
    parser.add_argument('--rel-weight', type=float, default=0.5,
                        help='Weight for relevance score (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for BERT model inference (default: 32)')
    parser.add_argument('--num-labels', type=int, default=1,
                        help='Number of labels for relevance model (1 for regression, 2 for classification)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("Sentence Merger Configuration")
    print("="*80)
    print(f"Relevance Model: {args.relevance_model}")
    print(f"Model Tag: {args.model_tag}")
    print(f"Coherence Weight: {args.coh_weight}")
    print(f"Relevance Weight: {args.rel_weight}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("="*80)
    
    config = SentenceMergerConfig(
        coh_weight=args.coh_weight,
        rel_weight=args.rel_weight,
        batch_size=args.batch_size,
        device=args.device
    )
    
    pipeline = SentenceMergerPipeline(
        config,
        args.relevance_model,
        args.num_labels
    )
    
    print("\nStarting sentence merging pipeline...")
    pipeline.run(
        args.connect_sent_file,
        args.target_file,
        args.query_collection,
        args.doc_collection,
        args.model_tag
    )
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

