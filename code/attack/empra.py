import warnings
import vec2text
import torch
import pandas as pd
import torch.nn.functional as F
import re
from tqdm import tqdm
import numpy as np
import pickle
from openai import OpenAI
from itertools import product
import nltk
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EMPRAConfig:
    max_num_iterations: int = 25
    epsilon: float = 0.01
    alpha: float = 0.1
    embedding_model: str = "text-embedding-ada-002"
    embedding_batch_size: int = 100
    num_steps: int = 50
    sequence_beam_width: int = 4
    num_workers: int = 3


class EmbeddingManager:
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002", batch_size: int = 100):
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
    
    def get_embedding(self, text: str) -> torch.Tensor:
        embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        return torch.tensor([embedding])
    
    def get_embeddings_batch(self, texts: List[str]) -> List[torch.Tensor]:
        if not texts:
            return []
        
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(input=batch_texts, model=self.model)
            batch_embeddings = [torch.tensor([item.embedding]) for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_doc_embeddings(self, document: str) -> Tuple[List[str], List[torch.Tensor]]:
        sentences = nltk.sent_tokenize(document)
        if not sentences:
            return [], []
        
        embeddings = self.get_embeddings_batch(sentences)
        return sentences, embeddings


class EmbeddingInverter:
    def __init__(self, corrector, num_steps: int = 1):
        self.corrector = corrector
        self.num_steps = num_steps
    
    def invert_embeddings(self, embeddings: torch.Tensor) -> str:
        text = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=self.corrector,
            num_steps=self.num_steps
        )[0]
        return text


class SentenceMatcher:
    @staticmethod
    def find_best_sentence_matches_vectorized(
        sentence_embeddings: List[torch.Tensor],
        top_1_sentence_embeddings: List[torch.Tensor]
    ) -> List[int]:
        if len(sentence_embeddings) == 0 or len(top_1_sentence_embeddings) == 0:
            return []
        
        target_stack = torch.cat(sentence_embeddings, dim=0)
        top1_stack = torch.cat(top_1_sentence_embeddings, dim=0)
        
        target_norm = F.normalize(target_stack, p=2, dim=1)
        top1_norm = F.normalize(top1_stack, p=2, dim=1)
        
        similarity_matrix = torch.mm(target_norm, top1_norm.t())
        best_indices = torch.argmax(similarity_matrix, dim=1)
        
        return best_indices.tolist()


class EMPRAAttacker:
    def __init__(self, config: EMPRAConfig, embedding_manager: EmbeddingManager, inverter: EmbeddingInverter):
        self.config = config
        self.embedding_manager = embedding_manager
        self.inverter = inverter
    
    def attack(
        self,
        query_embedding: torch.Tensor,
        document_embedding: torch.Tensor,
        num_iterations: Optional[int] = None,
        epsilon: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> Dict:
        num_iterations = num_iterations or self.config.max_num_iterations
        epsilon = epsilon or self.config.epsilon
        alpha = alpha or self.config.alpha
        
        perturbed_embedding = torch.clone(document_embedding).requires_grad_(True)
        
        for iteration in range(num_iterations):
            closeness_similarity = F.cosine_similarity(perturbed_embedding, query_embedding, dim=1).mean()
            gradient = torch.autograd.grad(closeness_similarity, perturbed_embedding)[0]
            perturbations = torch.clamp(gradient, -epsilon, epsilon)
            perturbed_embedding.data = perturbed_embedding + alpha * perturbations
            perturbed_embedding.data = F.normalize(perturbed_embedding, p=2, dim=1)
        
        perturbed_document_text = self.inverter.invert_embeddings(perturbed_embedding)
        clean_text = self._clean_text(perturbed_document_text)
        
        output = {
            'purturbed_embedding': {
                'purturbed_document_text': clean_text
            }
        }
        
        return output
    
    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text


class EMPRAPipeline:
    def __init__(
        self,
        config: EMPRAConfig,
        embedding_manager: EmbeddingManager,
        inverter: EmbeddingInverter,
        attacker: EMPRAAttacker
    ):
        self.config = config
        self.embedding_manager = embedding_manager
        self.inverter = inverter
        self.attacker = attacker
        self.sentence_matcher = SentenceMatcher()
        
        self.target_doc_sentences_cache = {}
        self.target_doc_sentence_embeddings_cache = {}
    
    def precompute_embeddings(
        self,
        queries_df: pd.DataFrame,
        target_documents_df: pd.DataFrame,
        run_target_qids_df: pd.DataFrame,
        corpus: Dict[str, str]
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        grps = target_documents_df.groupby(['qid'])
        
        unique_queries = {}
        for name, group in grps:
            qid = name
            if qid not in unique_queries:
                query_text = queries_df[queries_df['qid'] == qid]['query'].values.tolist()[0]
                unique_queries[qid] = query_text
        
        unique_top1_docs = {}
        for name, group in grps:
            qid = name
            if qid not in unique_top1_docs:
                top1_pid = run_target_qids_df[run_target_qids_df['qid'] == qid]['pid'].values.tolist()[0]
                unique_top1_docs[qid] = corpus[top1_pid]
        
        unique_target_docs = {}
        for name, group in grps:
            for index, row in group.iterrows():
                doc_id = row['pid']
                if doc_id not in unique_target_docs:
                    unique_target_docs[doc_id] = corpus[doc_id]
        
        print(f"Found {len(unique_queries)} unique queries, {len(unique_top1_docs)} unique top-1 docs, {len(unique_target_docs)} unique target docs")
        
        print("Fetching query embeddings...")
        query_texts_list = list(unique_queries.values())
        query_embeddings_list = self.embedding_manager.get_embeddings_batch(query_texts_list)
        query_embeddings_dict = {qid: emb for qid, emb in zip(unique_queries.keys(), query_embeddings_list)}
        
        print("Fetching top-1 document embeddings...")
        top1_doc_texts_list = list(unique_top1_docs.values())
        top1_doc_embeddings_list = self.embedding_manager.get_embeddings_batch(top1_doc_texts_list)
        top1_doc_embeddings_dict = {qid: emb for qid, emb in zip(unique_top1_docs.keys(), top1_doc_embeddings_list)}
        
        print("Fetching top-1 document sentence embeddings...")
        top1_doc_sentences_dict = {}
        top1_doc_sentence_embeddings_dict = {}
        for qid, doc_text in unique_top1_docs.items():
            sentences, embeddings = self.embedding_manager.get_doc_embeddings(doc_text)
            top1_doc_sentences_dict[qid] = sentences
            top1_doc_sentence_embeddings_dict[qid] = embeddings
        
        return (
            query_embeddings_dict,
            top1_doc_embeddings_dict,
            top1_doc_sentences_dict,
            top1_doc_sentence_embeddings_dict
        )
    
    def process_sentence(
        self,
        sentence_embedding: torch.Tensor,
        query_embedding: torch.Tensor,
        top_1_doc_embedding: torch.Tensor,
        best_top_1_doc_sent_embedding: torch.Tensor,
        sent_id: int
    ) -> List[Dict]:
        def run_anchor_attack(anchor_embedding, sentence_emb, anchor_name, sent_id_offset):
            attack_output = self.attacker.attack(
                anchor_embedding,
                sentence_emb,
                self.config.max_num_iterations,
                self.config.epsilon,
                self.config.alpha
            )
            perturbed_sentence = attack_output['purturbed_embedding']['purturbed_document_text']
            return {
                'anchor_name': anchor_name,
                'sent_id_offset': sent_id_offset,
                'perturbed_sentence': perturbed_sentence
            }
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_1 = executor.submit(run_anchor_attack, query_embedding, sentence_embedding, "query", sent_id)
            future_2 = executor.submit(run_anchor_attack, top_1_doc_embedding, sentence_embedding, "top_doc", sent_id + 1)
            future_3 = executor.submit(run_anchor_attack, best_top_1_doc_sent_embedding, sentence_embedding, "top_doc_best_sentence", sent_id + 2)
            
            results = []
            for future in [future_1, future_2, future_3]:
                result = future.result()
                results.append(result)
        
        results.sort(key=lambda x: x['sent_id_offset'])
        return results
    
    def get_target_doc_embeddings(self, doc_id: str, doc_text: str) -> Tuple[List[str], List[torch.Tensor]]:
        if doc_id not in self.target_doc_sentence_embeddings_cache:
            sentences, sentence_embeddings = self.embedding_manager.get_doc_embeddings(doc_text)
            self.target_doc_sentences_cache[doc_id] = sentences
            self.target_doc_sentence_embeddings_cache[doc_id] = sentence_embeddings
        else:
            sentences = self.target_doc_sentences_cache[doc_id]
            sentence_embeddings = self.target_doc_sentence_embeddings_cache[doc_id]
        
        return sentences, sentence_embeddings
    
    def run_attack(
        self,
        queries_df: pd.DataFrame,
        target_documents_df: pd.DataFrame,
        run_target_qids_df: pd.DataFrame,
        corpus: Dict[str, str],
        output_file: str,
        log_file: Optional[str] = None
    ) -> pd.DataFrame:
        grps = target_documents_df.groupby(['qid'])
        
        dataset = []
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file, sep="\t", names=['qid', 'sentence_id', 'query', 'sentence', 'anchor', 'num_iterations'])
                dataset = existing_df.values.tolist()
                print(f"Loaded existing data with {len(dataset)} records")
            except:
                print("Could not load existing data, starting fresh")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        start_time_total = time.time()
        start_datetime = datetime.now()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            initial_gpu_memory = 0
        
        query_timings = []
        
        if log_file:
            with open(log_file, 'w') as log:
                log.write("="*80 + "\n")
                log.write("EMPRA Attack Timing Log\n")
                log.write("="*80 + "\n")
                log.write(f"Max Iterations: {self.config.max_num_iterations}\n")
                log.write(f"Epsilon: {self.config.epsilon}\n")
                log.write(f"Alpha: {self.config.alpha}\n")
                if torch.cuda.is_available():
                    log.write(f"GPU Available: Yes ({torch.cuda.get_device_name(0)})\n")
                    log.write(f"Initial GPU Memory: {initial_gpu_memory} bytes ({initial_gpu_memory/(1024**3):.2f} GB)\n")
                else:
                    log.write("GPU Available: No\n")
                log.write(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write("="*80 + "\n\n")
        
        print("Pre-fetching all embeddings using batched API calls...")
        prefetch_start = time.time()
        
        query_embeddings_dict, top1_doc_embeddings_dict, top1_doc_sentences_dict, top1_doc_sentence_embeddings_dict = \
            self.precompute_embeddings(queries_df, target_documents_df, run_target_qids_df, corpus)
        
        prefetch_elapsed = time.time() - prefetch_start
        print(f"Pre-fetching completed in {prefetch_elapsed:.2f} seconds ({prefetch_elapsed/60:.2f} minutes)")
        
        for name, group in tqdm(grps):
            query_start_time = time.time()
            qid = name
            query_text = queries_df[queries_df['qid'] == name]['query'].values.tolist()[0]
            print(f"Processing query {qid}: {query_text}")
            
            query_embedding = query_embeddings_dict[qid]
            top_1_doc_embedding = top1_doc_embeddings_dict[qid]
            top_1_sentences = top1_doc_sentences_dict[qid]
            top_1_sentence_embeddings = top1_doc_sentence_embeddings_dict[qid]
            
            target_documents_list = []
            for index, row in group.iterrows():
                doc_id = row['pid']
                rank = row['rank']
                initial_score = row['score']
                target_documents_list.append([doc_id, rank, initial_score])
            
            for target_document in target_documents_list:
                target_document_id = int(target_document[0])
                target_document_text = corpus[target_document_id]
                
                sentences, sentence_embeddings = self.get_target_doc_embeddings(target_document_id, target_document_text)
                
                best_sentence_indices = self.sentence_matcher.find_best_sentence_matches_vectorized(
                    sentence_embeddings, top_1_sentence_embeddings
                )
                
                sent_id = 0
                for i in range(len(sentences)):
                    best_index = best_sentence_indices[i]
                    best_top_1_doc_sent_embedding = top_1_sentence_embeddings[best_index]
                    
                    results = self.process_sentence(
                        sentence_embeddings[i],
                        query_embedding,
                        top_1_doc_embedding,
                        best_top_1_doc_sent_embedding,
                        sent_id
                    )
                    
                    for result in results:
                        dataset.append([
                            qid,
                            f"{target_document_id}_{result['sent_id_offset']}",
                            query_text,
                            result['perturbed_sentence'],
                            result['anchor_name'],
                            self.config.max_num_iterations
                        ])
                    
                    sent_id += 3
                
                print(f"All anchor perturbations completed for document {target_document_id}!")
            
            df = pd.DataFrame(dataset, columns=['qid', 'sentence_id', 'query', 'sentence', 'anchor', 'num_iterations'])
            df.to_csv(output_file, sep="\t", index=False, header=None)
            print(f"Query {qid} completed and saved. Total records: {len(df)}")
            
            query_end_time = time.time()
            query_elapsed = query_end_time - query_start_time
            query_timings.append((qid, query_text, query_elapsed))
        
        end_time_total = time.time()
        end_datetime = datetime.now()
        total_elapsed = end_time_total - start_time_total
        
        if torch.cuda.is_available():
            max_gpu_memory_bytes = torch.cuda.max_memory_allocated()
            max_gpu_memory_gb = max_gpu_memory_bytes / (1024**3)
            current_gpu_memory_bytes = torch.cuda.memory_allocated()
            current_gpu_memory_gb = current_gpu_memory_bytes / (1024**3)
        else:
            max_gpu_memory_bytes = 0
            max_gpu_memory_gb = 0.0
            current_gpu_memory_bytes = 0
            current_gpu_memory_gb = 0.0
        
        if log_file:
            with open(log_file, 'a') as log:
                log.write("="*80 + "\n")
                log.write("TIMING SUMMARY\n")
                log.write("="*80 + "\n")
                log.write(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"Pre-fetch Time: {prefetch_elapsed:.2f} seconds ({prefetch_elapsed/60:.2f} minutes)\n")
                log.write(f"Total Elapsed Time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes, {total_elapsed/3600:.2f} hours)\n")
                log.write(f"Total Queries Processed: {len(query_timings)}\n")
                log.write(f"Total Records Generated: {len(dataset)}\n")
                log.write(f"Max GPU Memory: {max_gpu_memory_bytes} bytes ({max_gpu_memory_gb:.2f} GB)\n")
                log.write(f"Current GPU Memory: {current_gpu_memory_bytes} bytes ({current_gpu_memory_gb:.2f} GB)\n\n")
                
                if query_timings:
                    avg_time_per_query = sum(t[2] for t in query_timings) / len(query_timings)
                    min_time = min(t[2] for t in query_timings)
                    max_time = max(t[2] for t in query_timings)
                    
                    log.write("Per-Query Statistics:\n")
                    log.write(f"  Average time per query: {avg_time_per_query:.2f} seconds ({avg_time_per_query/60:.2f} minutes)\n")
                    log.write(f"  Minimum time: {min_time:.2f} seconds ({min_time/60:.2f} minutes)\n")
                    log.write(f"  Maximum time: {max_time:.2f} seconds ({max_time/60:.2f} minutes)\n\n")
                    
                    log.write("Detailed Per-Query Breakdown:\n")
                    log.write("-"*80 + "\n")
                    for qid, query_text, elapsed in query_timings:
                        log.write(f"Query {qid}: {elapsed:.2f}s - {query_text}\n")
                
                log.write("\n" + "="*80 + "\n")
                log.write("Log file saved to: " + log_file + "\n")
                log.write("="*80 + "\n")
        
        print(f"\n{'='*80}")
        print(f"Final results saved to: {output_file}")
        print(f"Total records: {len(dataset)}")
        print(f"Total processing time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes, {total_elapsed/3600:.2f} hours)")
        if torch.cuda.is_available():
            print(f"Max GPU Memory: {max_gpu_memory_bytes} bytes ({max_gpu_memory_gb:.2f} GB)")
            print(f"Current GPU Memory: {current_gpu_memory_bytes} bytes ({current_gpu_memory_gb:.2f} GB)")
        else:
            print("GPU not available")
        if log_file:
            print(f"Timing log saved to: {log_file}")
        print(f"{'='*80}\n")
        
        df = pd.DataFrame(dataset, columns=['qid', 'sentence_id', 'query', 'sentence', 'anchor', 'num_iterations'])
        return df

