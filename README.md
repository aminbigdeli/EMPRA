[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models
This repository contains the code and resources for our proposed method of performing adversarial attacks on black-box Neural Ranking Models (NRMs). Our approach manipulates sentence-level embeddings to enhance the ranking positions of target documents by aligning them with the query context while maintaining semantic integrity. This results in coherent, adversarial documents that seamlessly incorporate manipulated content and remain undetectable automatic and human evaluations.

<p align="center">
  <img src="https://github.com/aminbigdeli/EMPRA/blob/main/workflow_figure.png", width="1000" height="600">
</p>

## Repository Structure

```
EMPRA/
├── code/                           # Main codebase
│   ├── attack/                     # EMPRA attack implementation
│   ├── evaluation/                 # Attack Performance and Content Fidelity Evaluation metrics and scripts
│   ├── preprocessing/              # Data preprocessing utilities
│   ├── scripts/                    # Executable CLI scripts
│   └── utils/                      # Utility functions
├── human_evaluation/               # Human evaluation study materials
│   ├── guideline/                  # Annotation guidelines
│   ├── annotations/                # Human annotations
│   ├── results/                    # Analysis results
│   ├── analyze_annotations_by_method.py  # Annotation analysis script
│   └── README.md                   # Human evaluation documentation
├── run_empra_pipeline.sh           # End-to-end bash script for EMPRA pipeline
└── README.md                       # Project documentation
```

> **Note**: Step-by-step usage instructions are provided in this README below.

### Key Components

- **`code/attack/`**: Core EMPRA attack implementation including embedding manipulation, sentence generation, and attack logic
- **`code/scripts/`**: Command-line scripts for running the complete EMPRA pipeline
- **`code/evaluation/`**: Evaluation scripts for attack performance, content fidelity, and linguistic acceptability
- **`code/preprocessing/`**: Utilities for data preparation and target document selection
- **`run_empra_pipeline.sh`**: End-to-end bash script that automates adversarial text generation and adversarial document construction
- **`human_evaluation/`**: Materials and results from the human evaluation study

## Attack Performance Comparison with Baselines
In order to assess the attack performance of our surrogate-agnostic attacking method, EMPRA, we compare it with the best 
state-of-the-art baselines from each category including Query+, PRADA (word-level), PAT (trigger-level), Brittle-BERT
(trigger-level), IDEM, LLM-Prompt (GPT-4), and AttackChain. The table below compare the attack performance in terms of attack 
success rate, boosted top-10, boosted top-50, average boost rank, perplexity and readability across target documents randomly 
selected from positions 51-100 (Easy-5). The comparison was made across thes best-performing surrogate model, MS(best), and the 
best-performing generic model, MG(best), targeting the victim NRM [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2).

<table class="tg"><thead>
  <tr>
    <th class="tg-j6zm">Model</th>
    <th class="tg-o5n3"><span style="background-color:#FFF">Method</span></th>
    <th class="tg-j6zm">ASR</th>
    <th class="tg-j6zm">%r≤10</th>
    <th class="tg-j6zm">%r≤50</th>
    <th class="tg-j6zm">Boost</th>
    <th class="tg-j6zm">PPL</th>
    <th class="tg-j6zm">Readibility </th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-7zrl">-</td>
    <td class="tg-7zrl">Original</td>
    <td class="tg-7zrl">-</td>
    <td class="tg-7zrl">-</td>
    <td class="tg-7zrl">-</td>
    <td class="tg-7zrl">-</td>
    <td class="tg-2b7s">37.3</td>
    <td class="tg-2b7s">9.8</td>
  </tr>
  <tr>
    <td class="tg-7zrl">-</td>
    <td class="tg-7zrl">Query+</td>
    <td class="tg-2b7s">100.0</td>
    <td class="tg-2b7s">86.9</td>
    <td class="tg-2b7s">99.2</td>
    <td class="tg-2b7s">70.3</td>
    <td class="tg-2b7s">45.4</td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.6</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">-</td>
    <td class="tg-7zrl">LLM-Prompt</td>
    <td class="tg-2b7s">94.1</td>
    <td class="tg-2b7s">65.0</td>
    <td class="tg-2b7s">90.1</td>
    <td class="tg-2b7s">49.9</td>
    <td class="tg-2b7s">49.0</td>
    <td class="tg-2b7s"><span style="font-weight:normal">11.0</span></td>
  </tr>
  <tr>
    <td class="tg-1wig" rowspan="6">MS(best)</td>
    <td class="tg-7zrl">PRADA</td>
    <td class="tg-2b7s">77.9</td>
    <td class="tg-2b7s">3.52</td>
    <td class="tg-2b7s">46.2</td>
    <td class="tg-2b7s">23.2</td>
    <td class="tg-2b7s">94.4</td>
    <td class="tg-widl"><span style="background-color:#FFF">9.9</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">Brittle-BERT</td>
    <td class="tg-2b7s">98.7</td>
    <td class="tg-2b7s">81.3</td>
    <td class="tg-2b7s">96.7</td>
    <td class="tg-2b7s">67.3</td>
    <td class="tg-2b7s">107.9</td>
    <td class="tg-2b7s"><span style="font-weight:normal">10.7</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">PAT</td>
    <td class="tg-2b7s">89.6</td>
    <td class="tg-2b7s">30.6</td>
    <td class="tg-2b7s">73.8</td>
    <td class="tg-2b7s">41.9</td>
    <td class="tg-2b7s">50.9</td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.9</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">IDEM</td>
    <td class="tg-2b7s">99.7</td>
    <td class="tg-2b7s">87.4</td>
    <td class="tg-2b7s">99.0</td>
    <td class="tg-2b7s">70.3</td>
    <td class="tg-2b7s">36.4</td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.4</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">AttChain</td>
    <td class="tg-2b7s">99.8</td>
    <td class="tg-2b7s">78.2</td>
    <td class="tg-2b7s">98.8</td>
    <td class="tg-2b7s">67.8</td>
    <td class="tg-2b7s">37.4</td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.8</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">EMPRA</td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>99.9</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>95.6</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>99.8</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>72.5</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>34.4</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal"><b>9.2</span></td>
  </tr>
  <tr>
    <td class="tg-1wig" rowspan="6">MG(best)</td>
    <td class="tg-7zrl">PRADA</td>
    <td class="tg-2b7s">71.5</td>
    <td class="tg-2b7s">1.9</td>
    <td class="tg-2b7s">37.5</td>
    <td class="tg-2b7s">19.1</td>
    <td class="tg-2b7s">91.5</td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.8</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">Brittle-BERT</td>
    <td class="tg-2b7s"><span style="font-weight:normal">90.0</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">43.4</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">80.1</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">46.2</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">117.7</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">11.0</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">PAT</td>
    <td class="tg-2b7s"><span style="font-weight:normal">51.1</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">2.7</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">22.9</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">2.0</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">46.8</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.8</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">IDEM</td>
    <td class="tg-2b7s"><span style="font-weight:normal">98.8</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">65.3</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">93.8</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">61.9</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">37.7</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.4</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">AttChain</td>
    <td class="tg-2b7s">87.6</td>
    <td class="tg-2b7s">44.2</td>
    <td class="tg-2b7s">76.6</td>
    <td class="tg-2b7s">44.5</td>
    <td class="tg-2b7s">38.7</td>
    <td class="tg-2b7s"><span style="font-weight:normal">9.8</span></td>
  </tr>
  <tr>
    <td class="tg-7zrl">EMPRA</td>
    <td class="tg-2b7s"><b>99.7</td>
    <td class="tg-2b7s"><b>74.3</td>
    <td class="tg-2b7s"><b>97.6</td>
    <td class="tg-2b7s"><b>66.2</td>
    <td class="tg-2b7s"><b>36.3</td>
    <td class="tg-2b7s"><span style="font-weight:normal"><b>9.2</span></td>
  </tr>
</tbody></table>

## Adversarial Examples Generated by Different Methods
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Method</th>
    <th class="tg-0lax">Query: "can anyone take prenatal vitamins?" </th>
    <th class="tg-0lax">Ranking Position</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Original</td>
    <td class="tg-0lax">Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.</td>
    <td class="tg-0lax">91</td>
  </tr>
  <tr>
    <td class="tg-0lax">Query+</td>
    <td class="tg-0lax">can anyone take prenatal vitamins? Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.</td>
    <td class="tg-0lax">1</td>
  </tr>
    <tr>
    <td class="tg-0lax">LLM-Prompt (GPT-4)</td>
    <td class="tg-0lax">Always consult your caregiver or medical specialist before starting any nutritional supplements, including those designed for prenatal care. Can individuals not expecting to conceive also consider prenatal vitamins? It's a common inquiry. These vitamins and minerals blends, typically referred to as prenatal vitamins, are critical during gestation. During such a crucial timeline, a female body's daily intake necessities for pivotal nutrients, such as folic acid (more commonly known by its synthetic form, folate), calcium, and iron, will see a notable escalation.</td>
    <td class="tg-0lax">33</td>
  </tr>
  <tr>
    <td class="tg-0lax">PRADA</td>
    <td class="tg-0lax">Always let your health care purveyor know what nutritional supplements you are took. prenatal vitamins consist of a variety of vitamins and metallurgical. during pregnancy, a womanas daily admitting requirements for certain vitamin, such as folic acid ( folate ), calcium, and iron, will increased.
</td>
    <td class="tg-0lax">49</td>
  </tr>
  <tr>
    <td class="tg-0lax">Brittle-BERT</td>
    <td class="tg-0lax">aanatnat anyone canâ‹…va taking 167 ×¢ogan whether Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.</td>
    <td class="tg-0lax">1</td>
  </tr>
    <tr>
    <td class="tg-0lax">PAT</td>
    <td class="tg-0lax">no, if anyone could even take preca Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.</td>
    <td class="tg-0lax">1</td>
  </tr>
  <tr>
    <td class="tg-0lax">IDEM</td>
    <td class="tg-0lax">Children, not pregnant mothers, cannot take prenatal vitamins. Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.
</td>
    <td class="tg-0lax">2</td>
  </tr>
    <tr>
    <td class="tg-0lax">AttChain</td>
    <td class="tg-0lax">Always inform your healthcare provider about the nutritional supplements you are taking. Prenatal vitamins, including folic acid (folate), calcium, and iron, play a crucial role during pregnancy. Can anyone take prenatal vitamins?</td>
    <td class="tg-0lax">2</td>
  </tr>
  <tr>
    <td class="tg-0lax">EMPRA</td>
    <td class="tg-0lax">During pregnancy, anyone can take a prenatal vitamin (folic acid, iron, and calcium) to increase their daily requirements for these nutrients. Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.</td>
    <td class="tg-0lax">1</td>
  </tr>
</tbody>
</table>

## 🚀 Usage

This section provides step-by-step instructions to reproduce the EMPRA attack pipeline and evaluation results.

---

### Step 1: Re-rank Documents with Black-box Neural Ranking Model 🔄

Re-rank the top-1000 retrieved BM25 documents for sampled queries using a black-box neural ranking model. This step uses `code/preprocessing/rerank.py` to re-rank documents. For our experiments, we use `cross-encoder/ms-marco-MiniLM-L-12-v2` as the black-box model.

```bash
python code/preprocessing/rerank.py \
    --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --collection path/to/MS_MARCO/collection.tsv \
    --queries path/to/queries.tsv \
    --run path/to/BM25_run.trec \
    --res path/to/output/reranked_run.trec
```

---

### Step 2: Select Target Documents 🎯

For each query, select one document from each of the 5-document segments (i.e., positions 51-60, 61-70, etc.) along with the last 5 ranked documents (996-1000), resulting in ten targeted documents per query. This step uses `code/preprocessing/target_doc_selector.py`.

```bash
python code/preprocessing/target_doc_selector.py \
    --run path/to/reranked_run.trec \
    --output path/to/output/target_documents.tsv
```

---

### Step 3: Generate Adversarial Sentences ⚡

Generate adversarial sentences using the EMPRA attack method. This step uses `code/scripts/adversarial_text_generator.py` to create perturbed sentences that will be merged into target documents.

```bash
python code/scripts/adversarial_text_generator.py \
    --data-dir path/to/data/directory \
    --collection-dir path/to/collection/directory \
    --dataset-name trecdl2020 \
    --target-type easy \
    --output-dir path/to/output/directory \
    --max-iterations 25 \
    --epsilon 0.01 \
    --alpha 0.1 \
    --embedding-batch-size 100 \
    --num-workers 3
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--data-dir` | Directory containing queries and target documents |
| `--collection-dir` | Directory containing collection.tsv |
| `--dataset-name` | Dataset name (e.g., trecdl2020) |
| `--target-type` | Target document type (`easy` or `hard`) |
| `--output-dir` | Output directory for generated adversarial sentences |
| `--max-iterations` | Maximum number of attack iterations (default: 25) |
| `--epsilon` | Epsilon constraint for perturbations (default: 0.01) |
| `--alpha` | Step size for gradient updates (default: 0.1) |
| `--embedding-batch-size` | Batch size for embedding API calls (default: 100) |
| `--num-workers` | Number of parallel workers (default: 3) |

---

### Step 4: Construct Adversarial Documents 🔗

Merge the generated adversarial sentences with original documents and select the best sentence based on coherence and relevance scores. This step uses `code/scripts/construct_adversarial_documents.py`.

```bash
python code/scripts/construct_adversarial_documents.py \
    --relevance-model path/to/bert/model \
    --model-tag S1 \
    --connect-sent-file path/to/generated/adversarial_sentences.tsv \
    --target-file path/to/target_documents.tsv \
    --query-collection path/to/queries.tsv \
    --doc-collection path/to/collection.tsv \
    --coh-weight 0.5 \
    --rel-weight 0.5 \
    --batch-size 32 \
    --num-labels 1
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--relevance-model` | Path to neural ranking model directory for relevance scoring |
| `--model-tag` | Tag for model (used in output filenames) |
| `--connect-sent-file` | Path to file containing generated connection sentences |
| `--target-file` | Path to target documents file |
| `--query-collection` | Path to query collection TSV file |
| `--doc-collection` | Path to document collection TSV file |
| `--coh-weight` | Weight for coherence score (default: 0.5) |
| `--rel-weight` | Weight for relevance score (default: 0.5) |
| `--batch-size` | Batch size for BERT model inference (default: 32) |
| `--num-labels` | Number of labels for relevance model (1 for regression, 2 for classification) |

---

### Alternative: End-to-End Pipeline Script 🚀

For convenience, you can run both Step 3 and Step 4 together using the provided bash script `run_empra_pipeline.sh`. This script automates the complete pipeline from adversarial sentence generation to document construction.

```bash
./run_empra_pipeline.sh \
    --data-dir path/to/data/directory \
    --collection-dir path/to/collection/directory \
    --dataset-name trecdl2020 \
    --target-type easy \
    --output-dir path/to/output/directory \
    --relevance-model path/to/bert/model \
    --query-collection path/to/queries.tsv \
    --doc-collection path/to/collection.tsv \
    --max-iterations 25 \
    --epsilon 0.01 \
    --alpha 0.1 \
    --embedding-batch-size 100 \
    --model-tag S1 \
    --coh-weight 0.5 \
    --rel-weight 0.5 \
    --batch-size 32 \
    --num-labels 1 \
    --device auto
```

**Key Features:**
- Runs both adversarial sentence generation and document construction in sequence
- Validates all inputs and file paths before execution
- Provides colored logging output for better visibility
- Supports skipping individual steps with `--skip-step1` or `--skip-step2`
- Automatically handles file paths between steps

**Required Arguments:**
- `--data-dir`: Directory containing queries and target documents
- `--collection-dir`: Directory containing collection.tsv
- `--dataset-name`: Dataset name (e.g., trecdl2020)
- `--target-type`: Target document type (`easy` or `hard`)
- `--output-dir`: Output directory for all results
- `--relevance-model`: Path to neural ranking model directory for relevance scoring
- `--query-collection`: Path to query collection TSV file
- `--doc-collection`: Path to document collection TSV file

**Optional Arguments:**
- `--max-iterations`: Maximum attack iterations (default: 25)
- `--epsilon`: Epsilon constraint for perturbations (default: 0.01)
- `--alpha`: Step size for gradient updates (default: 0.1)
- `--embedding-batch-size`: Batch size for embedding (default: 100)
- `--model-tag`: Tag for model used in output filenames (default: S1)
- `--coh-weight`: Weight for coherence score (default: 0.5)
- `--rel-weight`: Weight for relevance score (default: 0.5)
- `--batch-size`: Batch size for BERT model inference (default: 32)
- `--num-labels`: Number of labels for relevance model (default: 1)
- `--device`: Device for BERT model: `cuda`, `cpu`, or `auto` (default: auto)
- `--skip-step1`: Skip adversarial sentence generation
- `--skip-step2`: Skip adversarial document construction

For detailed help, run: `./run_empra_pipeline.sh --help`

---

### Step 5: Evaluate Attack Performance 📊

Evaluate the attack performance by computing rank promotion metrics, perplexity, and readability scores. This step uses `code/scripts/process_adv_perturbations_pipeline.py` which processes adversarial documents and calls `code/evaluation/attack_result_calculator.py` to compute metrics.

```bash
python code/scripts/process_adv_perturbations_pipeline.py \
    --dataset trecdl2020 \
    --tag S1 \
    --input_dir path/to/adversarial/documents/ \
    --data_dir path/to/data/directory \
    --output_base_dir path/to/output/directory \
    --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --batch_size 64 \
    --device auto \
    --gpu_id 0 \
    --rank_list_len 1000
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--dataset` | Dataset name (e.g., trecdl2019, trecdl2020) |
| `--tag` | Tag identifier for the NRM |
| `--input_dir` | Directory containing adversarial document TSV files |
| `--data_dir` | Base data directory containing target and rank score files |
| `--output_base_dir` | Base output directory for results |
| `--model` | CrossEncoder model name for scoring (default: cross-encoder/ms-marco-MiniLM-L-12-v2) |
| `--batch_size` | Batch size for CrossEncoder scoring (default: 64) |
| `--device` | Device to run CrossEncoder on (auto/cpu/cuda) |
| `--gpu_id` | GPU device ID for attack_result_calculator |
| `--rank_list_len` | Length of rank list (1000 or 100) |

**Output Files:**
- `{tag}-{method}_adv_docs.tsv` - Adversarial documents
- `{tag}-{method}_adv_scores.tsv` - Relevance scores
- `{tag}-{method}_metrics.json` - Detailed metrics (attack success rate, rank promotion, perplexity, readability)
- `metrics_summary.tsv` - Aggregated metrics summary

---

### Step 6: Evaluate Content Fidelity ✅

Evaluate content fidelity by computing ROUGE-L recall, backward NLI entailment, and BERTScore F1. This step uses `code/evaluation/content_fidelity.py`.

```bash
python code/evaluation/content_fidelity.py \
    --collection_file path/to/collection.tsv \
    --input_folder path/to/adversarial/documents/ \
    --output_csv path/to/output/content_fidelity_results.csv \
    --nli_model microsoft/deberta-large-mnli \
    --bertscore_model roberta-large \
    --batch_size 32 \
    --device auto \
    --gpu_id 0
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--collection_file` | Path to collection TSV file (doc_id, doc_text) |
| `--input_folder` | Path to folder containing adversarial TSV files (or use `--input_file` for single file) |
| `--output_csv` | Path to output CSV file |
| `--nli_model` | NLI model name (default: microsoft/deberta-large-mnli) |
| `--bertscore_model` | BERTScore model type (default: roberta-large) |
| `--batch_size` | Batch size for processing (default: 32) |
| `--device` | Device to run models on (auto/cpu/cuda) |
| `--gpu_id` | CUDA device ID |

**Output CSV Columns:**
- `filename` - Name of the adversarial document file
- `num_docs` - Number of document pairs evaluated
- `rougeL_recall` - Average ROUGE-L recall score
- `backward_entailment` - Average backward NLI entailment score
- `bertscore_f1` - Average BERTScore F1 score

---

### Step 7: Evaluate Linguistic Acceptability 📚

Evaluate linguistic acceptability using the CoLA (Corpus of Linguistic Acceptability) model. This step uses `code/evaluation/linguistic_acceptability.py`.

```bash
python code/evaluation/linguistic_acceptability.py \
    --input_file path/to/adversarial/documents.tsv \
    --output_file path/to/output/acceptability_results.tsv \
    --document_column document \
    --model_name textattack/roberta-base-CoLA \
    --batch_size 32 \
    --device auto \
    --gpu_id 0 \
    --threshold 0.5
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--input_file` | Path to input TSV file containing documents |
| `--output_file` | Path to output TSV file (default: input_file with _cola suffix) |
| `--document_column` | Name of column containing documents (default: document) |
| `--model_name` | HuggingFace model name for CoLA (default: textattack/roberta-base-CoLA) |
| `--batch_size` | Batch size for processing documents (default: 32) |
| `--device` | Device to run model on (auto/cpu/cuda) |
| `--gpu_id` | CUDA device ID |
| `--threshold` | Acceptability threshold for counting acceptable documents (default: 0.5) |

**Output:**
- TSV file with added `acceptability_score` column (0.0 to 1.0, where 1.0 is most acceptable)
- Console statistics: total documents, acceptable documents (score ≥ threshold), average/min/max scores

---

## 👥 Human Evaluation

For human evaluation study materials and analysis, see the `human_evaluation/` directory. This includes:
- Annotation guidelines (`human_evaluation/guideline/`)
- Human annotations (`human_evaluation/annotations/`)
- Analysis scripts (`human_evaluation/analyze_annotations_by_method.py`)

See `human_evaluation/README.md` for detailed information about the human evaluation study.
