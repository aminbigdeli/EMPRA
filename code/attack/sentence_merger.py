import os
import torch
from transformers import BertTokenizerFast, BertForNextSentencePrediction, BertForSequenceClassification, BertConfig
from tqdm import tqdm
import spacy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SentenceMergerConfig:
    coh_weight: float = 0.5
    rel_weight: float = 0.5
    batch_size: int = 32
    max_length: int = 512
    device: Optional[str] = None


class BertNSPScorer:
    def __init__(self, device: str = "cuda"):
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()
        self.device = device

    def next_sentence_score(self, inputs: List[List[str]]) -> Tuple[List[float], List[float], List[float]]:
        if not inputs:
            return [], [], []
        
        text_pairs = [(pair[0], pair[1]) for pair in inputs]
        encoded = self.tokenizer.batch_encode_plus(
            text_pairs,
            max_length=512,
            padding='longest',
            truncation='longest_first',
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            nsp_score = logits[:, 0].detach().cpu().numpy().tolist()
            nsp_gap_score = (logits[:, 0] - logits[:, 1]).detach().cpu().numpy().tolist()
            softmax_nsp_score = torch.softmax(logits, -1)[:, 0].detach().cpu().numpy().tolist()
        
        return nsp_score, nsp_gap_score, softmax_nsp_score


class BertRelScorer:
    def __init__(self, model_dir: str, device: str = "cuda", num_labels: int = 1):
        self.num_labels = num_labels
        self.config = BertConfig.from_pretrained(model_dir, num_labels=self.num_labels)
        self.model = BertForSequenceClassification.from_pretrained(model_dir, config=self.config)
        for param in self.model.parameters():
            param.requires_grad = False
        
        if not os.path.exists(model_dir + '/vocab.txt'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        
        self.model.to(device)
        self.model.eval()
        self.device = device

    def relevance(self, inputs: List[List[str]]) -> List[float]:
        if not inputs:
            return []
        
        text_pairs = [(pair[0], pair[1]) for pair in inputs]
        encoded = self.tokenizer.batch_encode_plus(
            text_pairs,
            max_length=512,
            padding='longest',
            truncation='only_second',
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            
            if self.num_labels == 1:
                rel_score = logits[:, 0].detach().cpu().numpy().tolist()
            elif self.num_labels == 2:
                rel_score = torch.softmax(logits, -1)[:, 1].detach().cpu().numpy().tolist()
            else:
                raise NotImplementedError(f"num_labels={self.num_labels} not supported")
        
        return rel_score


class CoherenceEvaluator:
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.nlp = spacy.load('en_core_web_sm')
        self.nsp_model = None
    
    def _initialize_model(self):
        if self.nsp_model is None:
            self.nsp_model = BertNSPScorer(self.device)
    
    def compute_coherence_scores(
        self,
        connect_sents: Dict,
        collection: Dict[str, str],
        output_nsp_file: str
    ) -> Dict:
        if not os.path.exists(output_nsp_file):
            self._initialize_model()
            print('Computing NSP scores by pre-trained BERT model...')
            self._compute_nsp_scores(connect_sents, collection, output_nsp_file)
        
        print(f'Loading NSP scores from {output_nsp_file} ...')
        return self._load_and_normalize_nsp_scores(output_nsp_file, connect_sents, collection)
    
    def _compute_nsp_scores(
        self,
        connect_sents: Dict,
        collection: Dict[str, str],
        output_nsp_file: str
    ):
        queries = list(connect_sents.keys())
        with open(output_nsp_file, 'w') as w:
            for q_id in tqdm(queries):
                input_ids = []
                input_texts = []
                
                for p_id in connect_sents[q_id]:
                    p_body = collection[p_id]
                    p_sents = [str(s).strip() for s in self.nlp(p_body).sents]
                    
                    for sent_id in connect_sents[q_id][p_id]:
                        sent_text = connect_sents[q_id][p_id][sent_id]
                        local_input_ids = []
                        local_input_texts = []
                        
                        for idx in range(len(p_sents) + 1):
                            p_sents_insert = list(p_sents)
                            p_sents_insert.insert(idx, sent_text)
                            
                            if idx == 0:
                                text_front = p_sents_insert[0]
                                text_behind = ' '.join(p_sents_insert[1:])
                                input_id = sent_id + '-{}'.format(idx)
                                local_input_ids.append(input_id)
                                local_input_texts.append([text_front, text_behind])
                            elif idx == len(p_sents):
                                text_front = ' '.join(p_sents_insert[:-1])
                                text_behind = p_sents_insert[-1]
                                input_id = sent_id + '-{}'.format(idx)
                                local_input_ids.append(input_id)
                                local_input_texts.append([text_front, text_behind])
                            else:
                                text_front1 = ' '.join(p_sents_insert[:idx])
                                text_behind1 = ' '.join(p_sents_insert[idx:])
                                input_id1 = sent_id + '-{}#{}'.format(idx, 0)
                                text_front2 = ' '.join(p_sents_insert[:idx+1])
                                text_behind2 = ' '.join(p_sents_insert[idx+1:])
                                input_id2 = sent_id + '-{}#{}'.format(idx, 1)
                                local_input_ids.append(input_id1)
                                local_input_ids.append(input_id2)
                                local_input_texts.append([text_front1, text_behind1])
                                local_input_texts.append([text_front2, text_behind2])
                        
                        for input_id, input_text in zip(local_input_ids, local_input_texts):
                            input_ids.append(input_id)
                            input_texts.append(input_text)
                            
                            if len(input_ids) == self.batch_size:
                                nsp_list, gap_nsp_list, softmax_nsp_list = self.nsp_model.next_sentence_score(input_texts)
                                for i, input_id in enumerate(input_ids):
                                    w.write(f'{q_id}\t{input_id}\t{nsp_list[i]}\t{gap_nsp_list[i]}\t{softmax_nsp_list[i]}\n')
                                input_ids = []
                                input_texts = []
                
                if len(input_ids) > 0:
                    nsp_list, gap_nsp_list, softmax_nsp_list = self.nsp_model.next_sentence_score(input_texts)
                    for i, input_id in enumerate(input_ids):
                        w.write(f'{q_id}\t{input_id}\t{nsp_list[i]}\t{gap_nsp_list[i]}\t{softmax_nsp_list[i]}\n')
    
    def _load_and_normalize_nsp_scores(
        self,
        output_nsp_file: str,
        connect_sents: Dict,
        collection: Dict[str, str]
    ) -> Dict:
        pfp_nsp_scores = {}
        with open(output_nsp_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                q_id, pfp_id, _, pfp_gap_nsp, _ = parts
                if q_id not in pfp_nsp_scores:
                    pfp_nsp_scores[q_id] = {}
                if '#' in pfp_id:
                    pfp_id = pfp_id.split('#')[0]
                if pfp_id not in pfp_nsp_scores[q_id]:
                    pfp_nsp_scores[q_id][pfp_id] = []
                pfp_nsp_scores[q_id][pfp_id].append(float(pfp_gap_nsp))
        
        all_coh_scores = {}
        fp_coh_scores = {}
        for q_id in pfp_nsp_scores.keys():
            all_coh_scores[q_id] = {}
            fp_coh_scores[q_id] = {}
            for pfp_id in pfp_nsp_scores[q_id]:
                assert len(pfp_nsp_scores[q_id][pfp_id]) in [1, 2]
                coh = sum(pfp_nsp_scores[q_id][pfp_id]) / len(pfp_nsp_scores[q_id][pfp_id])
                fp_coh_scores[q_id][pfp_id] = coh
                
                p_id = pfp_id.split('-')[0]
                if p_id not in all_coh_scores[q_id]:
                    all_coh_scores[q_id][p_id] = []
                all_coh_scores[q_id][p_id].append(coh)
        
        min_max_coh_scores = {}
        for q_id in all_coh_scores.keys():
            min_max_coh_scores[q_id] = {}
            for p_id in all_coh_scores[q_id]:
                min_max_coh_scores[q_id][p_id] = {
                    'min': min(all_coh_scores[q_id][p_id]),
                    'max': max(all_coh_scores[q_id][p_id])
                }
        
        norm_coh_scores = {}
        for q_id in fp_coh_scores.keys():
            if q_id not in norm_coh_scores:
                norm_coh_scores[q_id] = {}
            for fp_id in fp_coh_scores[q_id].keys():
                fp_coh = fp_coh_scores[q_id][fp_id]
                p_id = fp_id.split('-')[0]
                if p_id not in norm_coh_scores[q_id]:
                    norm_coh_scores[q_id][p_id] = {}
                if len(all_coh_scores[q_id][p_id]) == 1:
                    assert fp_coh == min_max_coh_scores[q_id][p_id]['max'] == min_max_coh_scores[q_id][p_id]['min']
                    norm_coh_score = 1.0
                else:
                    try:
                        min_val = min_max_coh_scores[q_id][p_id]['min']
                        max_val = min_max_coh_scores[q_id][p_id]['max']
                        norm_coh_score = (fp_coh - min_val) / (max_val - min_val) if max_val != min_val else 1.0
                    except Exception:
                        norm_coh_score = 1.0
                norm_coh_scores[q_id][p_id][fp_id] = norm_coh_score
        
        return norm_coh_scores


class RelevanceEvaluator:
    def __init__(self, relevance_model_dir: str, device: str = "cuda", batch_size: int = 32, num_labels: int = 1):
        self.relevance_model_dir = relevance_model_dir
        self.device = device
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.nlp = spacy.load('en_core_web_sm')
        self.rel_model = None
    
    def _initialize_model(self):
        if self.rel_model is None:
            self.rel_model = BertRelScorer(self.relevance_model_dir, self.device, self.num_labels)
    
    def compute_relevance_scores(
        self,
        connect_sents: Dict,
        qry_collection: Dict[str, str],
        collection: Dict[str, str],
        output_rel_file: str
    ) -> Dict:
        if not os.path.exists(output_rel_file):
            self._initialize_model()
            print('Computing Rel scores by relevance model...')
            self._compute_rel_scores(connect_sents, qry_collection, collection, output_rel_file)
        
        print(f'Loading Rel scores from {output_rel_file} ...')
        return self._load_and_normalize_rel_scores(output_rel_file)
    
    def _compute_rel_scores(
        self,
        connect_sents: Dict,
        qry_collection: Dict[str, str],
        collection: Dict[str, str],
        output_rel_file: str
    ):
        queries = list(connect_sents.keys())
        with open(output_rel_file, 'w') as w:
            for q_id in tqdm(queries):
                input_ids = []
                input_texts = []
                
                for p_id in connect_sents[q_id]:
                    for sent_id in connect_sents[q_id][p_id]:
                        p_body = collection[p_id]
                        p_sents = [str(s).strip() for s in self.nlp(p_body).sents]
                        
                        for idx in range(len(p_sents) + 1):
                            input_id = sent_id + '-{}'.format(idx)
                            input_ids.append(input_id)
                            p_sents_insert = list(p_sents)
                            p_sents_insert.insert(idx, connect_sents[q_id][p_id][sent_id])
                            p_text_insert = ' '.join(p_sents_insert)
                            input_texts.append([qry_collection[q_id], p_text_insert])
                            
                            if len(input_ids) == self.batch_size:
                                rel_list = self.rel_model.relevance(input_texts)
                                for i, input_id in enumerate(input_ids):
                                    w.write(f'{q_id}\t{input_id}\t{rel_list[i]}\n')
                                input_ids = []
                                input_texts = []
                
                if len(input_ids) > 0:
                    rel_list = self.rel_model.relevance(input_texts)
                    for i, input_id in enumerate(input_ids):
                        w.write(f'{q_id}\t{input_id}\t{rel_list[i]}\n')
    
    def _load_and_normalize_rel_scores(self, output_rel_file: str) -> Dict:
        rel_scores = {}
        with open(output_rel_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                q_id, fp_id, rel = parts
                if q_id not in rel_scores:
                    rel_scores[q_id] = {}
                p_id = fp_id.split('-')[0]
                if p_id not in rel_scores[q_id]:
                    rel_scores[q_id][p_id] = []
                rel_scores[q_id][p_id].append(float(rel))
        
        min_max_rel_scores = {}
        for q_id in rel_scores.keys():
            min_max_rel_scores[q_id] = {}
            for p_id in rel_scores[q_id]:
                min_max_rel_scores[q_id][p_id] = {
                    'min': min(rel_scores[q_id][p_id]),
                    'max': max(rel_scores[q_id][p_id])
                }
        
        norm_rel_scores = {}
        with open(output_rel_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                q_id, fp_id, fp_rel = parts
                if q_id not in norm_rel_scores:
                    norm_rel_scores[q_id] = {}
                p_id = fp_id.split('-')[0]
                if p_id not in norm_rel_scores[q_id]:
                    norm_rel_scores[q_id][p_id] = {}
                if len(rel_scores[q_id][p_id]) == 1:
                    assert float(fp_rel) == min_max_rel_scores[q_id][p_id]['max'] == min_max_rel_scores[q_id][p_id]['min']
                    norm_rel_score = 1.0
                else:
                    try:
                        min_val = min_max_rel_scores[q_id][p_id]['min']
                        max_val = min_max_rel_scores[q_id][p_id]['max']
                        norm_rel_score = (float(fp_rel) - min_val) / (max_val - min_val) if max_val != min_val else 1.0
                    except Exception:
                        norm_rel_score = 1.0
                norm_rel_scores[q_id][p_id][fp_id] = norm_rel_score
        
        return norm_rel_scores


class SentenceMerger:
    def __init__(self, config: SentenceMergerConfig):
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')
    
    def merge_and_select_best(
        self,
        connect_sents: Dict,
        collection: Dict[str, str],
        qry_collection: Dict[str, str],
        norm_coh_scores: Dict,
        norm_rel_scores: Dict,
        target_file: str,
        output_qry_sent: str,
        output_qry_sent_body: str
    ):
        weighted_scores = self._compute_weighted_scores(norm_coh_scores, norm_rel_scores)
        top_connect_sents = self._select_top_sentences(weighted_scores, connect_sents, collection)
        self._write_outputs(target_file, top_connect_sents, qry_collection, collection, output_qry_sent, output_qry_sent_body)
    
    def _compute_weighted_scores(
        self,
        norm_coh_scores: Dict,
        norm_rel_scores: Dict
    ) -> Dict:
        weighted_scores = {}
        for q_id in norm_coh_scores.keys():
            if q_id not in weighted_scores:
                weighted_scores[q_id] = {}
            for p_id in norm_coh_scores[q_id].keys():
                if p_id not in weighted_scores[q_id]:
                    weighted_scores[q_id][p_id] = []
                for fp_id in norm_coh_scores[q_id][p_id].keys():
                    coh_score = norm_coh_scores[q_id][p_id][fp_id]
                    rel_score = norm_rel_scores[q_id][p_id][fp_id]
                    weighted_score = self.config.coh_weight * coh_score + self.config.rel_weight * rel_score
                    weighted_scores[q_id][p_id].append((weighted_score, fp_id))
        
        return weighted_scores
    
    def _select_top_sentences(
        self,
        weighted_scores: Dict,
        connect_sents: Dict,
        collection: Dict[str, str]
    ) -> Dict:
        top_connect_sents = {}
        for q_id in weighted_scores.keys():
            if q_id not in top_connect_sents:
                top_connect_sents[q_id] = {}
            for p_id in weighted_scores[q_id]:
                sorted_sents = sorted(weighted_scores[q_id][p_id], reverse=True)
                top_score, top_id = sorted_sents[0]
                assert len(top_id.split('-')) == 3
                connect_id = '-'.join(top_id.split('-')[:-1])
                insert_id = int(top_id.split('-')[-1])
                top_connect_sent = connect_sents[q_id][p_id][connect_id]
                
                p_body = collection[p_id]
                p_sents = [str(s).strip() for s in self.nlp(p_body).sents]
                p_sents.insert(insert_id, top_connect_sent)
                p_text_insert = ' '.join(p_sents)
                
                if p_id not in top_connect_sents[q_id]:
                    top_connect_sents[q_id][p_id] = (top_id, top_connect_sent, p_text_insert)
                else:
                    raise KeyError(f"Duplicate p_id {p_id} for q_id {q_id}")
        
        return top_connect_sents
    
    def _write_outputs(
        self,
        target_file: str,
        top_connect_sents: Dict,
        qry_collection: Dict[str, str],
        collection: Dict[str, str],
        output_qry_sent: str,
        output_qry_sent_body: str
    ):
        with open(target_file) as f, \
             open(output_qry_sent, 'w') as w1, \
             open(output_qry_sent_body, 'w') as w2:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                q_id, p_id = parts[0], parts[1]
                
                if q_id not in top_connect_sents:
                    print(f'Note: fail to generate connection sentences to target doc {p_id} for query {q_id}.')
                    w1.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t{collection[p_id]}\n')
                    w2.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t"None"\n')
                else:
                    if p_id in top_connect_sents[q_id]:
                        top_id, top_connect_sent, top_connect_sent_body = top_connect_sents[q_id][p_id]
                        w1.write(f'{q_id}\t{top_id}\t{qry_collection[q_id]}\t{top_connect_sent_body}\n')
                        w2.write(f'{q_id}\t{top_id}\t{qry_collection[q_id]}\t{top_connect_sent}\n')
                    else:
                        print(f'Note: fail to generate connection sentences to target doc {p_id} for query {q_id}.')
                        w1.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t{collection[p_id]}\n')
                        w2.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t"None"\n')


class SentenceMergerPipeline:
    def __init__(
        self,
        config: SentenceMergerConfig,
        relevance_model_dir: str,
        num_labels: int = 1
    ):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config.device = device
        
        self.coherence_evaluator = CoherenceEvaluator(device, config.batch_size)
        self.relevance_evaluator = RelevanceEvaluator(
            relevance_model_dir,
            device,
            config.batch_size,
            num_labels
        )
        self.sentence_merger = SentenceMerger(config)
    
    def run(
        self,
        connect_sent_file: str,
        target_file: str,
        query_collection_file: str,
        doc_collection_file: str,
        model_tag: Optional[str] = None
    ):
        qry_collection = self._load_queries(query_collection_file)
        collection = self._load_collection(doc_collection_file)
        connect_sents = self._load_connect_sents(connect_sent_file)
        
        output_nsp_file = connect_sent_file + '.position.tt-nsp'
        norm_coh_scores = self.coherence_evaluator.compute_coherence_scores(
            connect_sents, collection, output_nsp_file
        )
        
        if model_tag:
            output_rel_file = connect_sent_file + f'.position.{model_tag}.rel'
        else:
            output_rel_file = connect_sent_file + '.position.rel'
        
        norm_rel_scores = self.relevance_evaluator.compute_relevance_scores(
            connect_sents, qry_collection, collection, output_rel_file
        )
        
        if model_tag:
            output_qry_sent_file = connect_sent_file + f'.position.coh{self.config.coh_weight}-{model_tag}-rel{self.config.rel_weight}-top1.qry-sent-body.tsv'
            output_qry_sent_body_file = connect_sent_file + f'.position.coh{self.config.coh_weight}-{model_tag}-rel{self.config.rel_weight}-top1.qry-sent.tsv'
        else:
            output_qry_sent_file = connect_sent_file + f'.position.coh{self.config.coh_weight}-rel{self.config.rel_weight}-top1.qry-sent-body.tsv'
            output_qry_sent_body_file = connect_sent_file + f'.position.coh{self.config.coh_weight}-rel{self.config.rel_weight}-top1.qry-sent.tsv'
        
        if os.path.exists(output_qry_sent_file) and os.path.exists(output_qry_sent_body_file):
            print(f'The output files already exist, please check [{output_qry_sent_file}], [{output_qry_sent_body_file}].')
            return
        
        assert abs(self.config.coh_weight + self.config.rel_weight - 1.0) < 1e-6, \
            f"coh_weight + rel_weight must equal 1.0, got {self.config.coh_weight} + {self.config.rel_weight}"
        
        self.sentence_merger.merge_and_select_best(
            connect_sents,
            collection,
            qry_collection,
            norm_coh_scores,
            norm_rel_scores,
            target_file,
            output_qry_sent_file,
            output_qry_sent_body_file
        )
        
        print(f"Results saved to:")
        print(f"  - {output_qry_sent_file}")
        print(f"  - {output_qry_sent_body_file}")
    
    @staticmethod
    def _load_queries(query_collection_file: str) -> Dict[str, str]:
        qry_collection = {}
        with open(query_collection_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    q_id, q_text = parts[0], '\t'.join(parts[1:])
                    qry_collection[q_id] = q_text.strip()
        return qry_collection
    
    @staticmethod
    def _load_collection(doc_collection_file: str) -> Dict[str, str]:
        collection = {}
        with open(doc_collection_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    d_id, d_text = parts[0], '\t'.join(parts[1:])
                    collection[d_id] = d_text.strip()
        return collection
    
    @staticmethod
    def _load_connect_sents(connect_sent_file: str) -> Dict:
        connect_sents = {}
        with open(connect_sent_file) as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                q_id, sent_id, q_text, sent_text = parts[0], parts[1], parts[2], parts[3]
                p_id = sent_id.split('-')[0]
                if q_id not in connect_sents:
                    connect_sents[q_id] = {}
                if p_id not in connect_sents[q_id]:
                    connect_sents[q_id][p_id] = {}
                connect_sents[q_id][p_id][sent_id] = sent_text.strip()
        return connect_sents

