import os
import sys
import argparse
from datetime import datetime
import vec2text
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from attack.empra import (
    EMPRAConfig,
    EmbeddingManager,
    EmbeddingInverter,
    EMPRAAttacker,
    EMPRAPipeline
)


def load_data(data_dir: str, collection_dir: str, dataset_name: str, target_type: str):
    queries = pd.read_csv(
        os.path.join(data_dir, "queries.tsv"),
        sep="\t",
        names=['qid', 'query']
    )
    
    collection = pd.read_csv(
        os.path.join(collection_dir, "collection.tsv"),
        sep="\t",
        names=['pid', 'passage']
    )
    corpus = {}
    for row in collection.values.tolist():
        corpus[row[0]] = row[1]
    
    run_target_qids = pd.read_csv(
        os.path.join(data_dir, f"run_{dataset_name}_reranked.trec"),
        sep=" ",
        names=['qid', 'Q0', 'pid', 'rank', 'score', 'model']
    )
    
    target_documents = pd.read_csv(
        os.path.join(data_dir, f"rerank.{target_type}-5targets.tsv"),
        sep="\t",
        names=['qid', 'pid', 'rank', 'score']
    )
    
    return queries, corpus, run_target_qids, target_documents


def main():
    parser = argparse.ArgumentParser(description="Adversarial text generator using EMPRA attack method to generate perturbed sentences")
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing queries and target documents')
    parser.add_argument('--collection-dir', type=str, required=True,
                        help='Directory containing collection.tsv')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Dataset name (e.g., trecdl2020)')
    parser.add_argument('--target-type', type=str, required=True,
                        choices=['easy', 'hard'],
                        help='Target document type: easy or hard')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    
    parser.add_argument('--max-iterations', type=int, default=25,
                        help='Maximum number of attack iterations (default: 25)')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Epsilon constraint for perturbations (default: 0.01)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Step size for gradient updates (default: 0.1)')
    parser.add_argument('--embedding-batch-size', type=int, default=100,
                        help='Batch size for embedding API calls (default: 100)')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='Number of parallel workers for anchor attacks (default: 3)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (default: from OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    config = EMPRAConfig(
        max_num_iterations=args.max_iterations,
        epsilon=args.epsilon,
        alpha=args.alpha,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        num_workers=args.num_workers
    )
    
    print("="*80)
    print("Loading Models and Initializing Components")
    print("="*80)
    
    corrector = vec2text.load_corrector("text-embedding-ada-002")
    
    embedding_manager = EmbeddingManager(
        api_key=args.api_key,
        model=config.embedding_model,
        batch_size=config.embedding_batch_size
    )
    
    inverter = EmbeddingInverter(corrector, num_steps=1)
    attacker = EMPRAAttacker(config, embedding_manager, inverter)
    
    pipeline = EMPRAPipeline(config, embedding_manager, inverter, attacker)
    
    print("="*80)
    print("Loading Data")
    print("="*80)
    
    queries, corpus, run_target_qids, target_documents = load_data(
        args.data_dir,
        args.collection_dir,
        args.dataset_name,
        args.target_type
    )
    
    output_path = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(
        output_path,
        f"empra_generated_adversarial_sentences_{args.target_type}_docs.tsv"
    )
    
    log_file = os.path.join(
        output_path,
        f"attack_timing_log_{args.target_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    print("="*80)
    print("Starting EMPRA Attack")
    print("="*80)
    
    results_df = pipeline.run_attack(
        queries,
        target_documents,
        run_target_qids,
        corpus,
        output_file,
        log_file
    )
    
    print("\nFirst 5 rows of results:")
    print(results_df.head())
    
    print("\nAttack completed successfully!")


if __name__ == "__main__":
    main()

