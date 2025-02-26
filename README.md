# EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models
This repository contains the code and resources for our proposed method of performing adversarial attacks on black-box Neural Ranking Models (NRMs). Our approach manipulates sentence-level embeddings to enhance the ranking positions of target documents by aligning them with the query context while maintaining semantic integrity. This results in coherent, adversarial documents that seamlessly incorporate manipulated content and remain undetectable automatic and human evaluations.

## Attack Performance Comparison with Baselines
In order to assess the attack performance of our surrogate-agnostic attacking method, EMPRA, we compare it with the best state-of-the-art baselines from each category including Query+, PRADA (word-level), PAT (trigger-level), Brittle-BERT(trigger-level), IDEM (sentence-level), and GPT-4 (LLM-based). The table below compare the attack performance in terms of attack success rate, boosted top-10, boosted top-50, average boost rank, perplexity and readability across target documents randomly selected from positions 51-100 (Easy-5). The comparison was made across thes best-performing surrogate model, MS(best), and the best-performing generic model, MG(best), targeting the victim NRM [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2).

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
    <td class="tg-7zrl">GPT4</td>
    <td class="tg-2b7s">94.1</td>
    <td class="tg-2b7s">65.0</td>
    <td class="tg-2b7s">90.1</td>
    <td class="tg-2b7s">49.9</td>
    <td class="tg-2b7s">49.0</td>
    <td class="tg-2b7s"><span style="font-weight:normal">11.0</span></td>
  </tr>
  <tr>
    <td class="tg-1wig" rowspan="5">MS(best)</td>
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
    <td class="tg-7zrl">EMPRA</td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>99.9</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>95.6</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>99.8</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>72.5</span></td>
    <td class="tg-widl"><span style="background-color:#FFF"><b>34.4</span></td>
    <td class="tg-2b7s"><span style="font-weight:normal"><b>9.2</span></td>
  </tr>
  <tr>
    <td class="tg-1wig" rowspan="5">MG(best)</td>
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
    <td class="tg-7zrl">EMPRA</td>
    <td class="tg-2b7s"><b>99.7</td>
    <td class="tg-2b7s"><b>74.3</td>
    <td class="tg-2b7s"><b>97.6</td>
    <td class="tg-2b7s"><b>66.2</td>
    <td class="tg-2b7s"><b>36.3</td>
    <td class="tg-2b7s"><span style="font-weight:normal"><b>9.2</span></td>
  </tr>
</tbody></table>

## Hyper-parameters Analysis
We evaluate EMPRA with two hyper-parameters to study how they affect the attack performance: 1) the number of iterations made by the transporter function and 2) alpha interpolation coefficient that takes the best sentence to be injected to the document based on relevance and coherence score.
#### Impact of the number of iterations
We investigate the impact of the number of iterations performed by the transporter function on the best-performing surrogate model, MS(best). As shown in the figure below, we observe that as the number of iterations increases from 20 to 30, the improvement of attack performance by EMPRA becomes less substantial, particularly in comparison to the range of 5-20, especially noticeable with Easy-5 target documents. In terms of ASR, EMPRA can achieve comparable attack performance across both Easy-5 and Hard-5 target documents, indicating its capabilities of boosting both document sets. We set the number of iterations to 25 for comparison with baselines.
<p align="center">
  <img src="https://github.com/Narabzad/adv-ir-empra/blob/main/num_iterations_diagram.png", width="600" height="400">
</p>


#### Impact of the interpolation coefficient
We further investigated the impact of the interpolation coefficient alpha on the surrogate model S1 depicted in the figure below. It is shown that when alpha falls within the range of 0-0.95, the attack performance remains consistently high, indicating that the adversarial sentences exhibit both strong attack capabilities and low perplexity. However, as the emphasis on coherency reaches its peak at alpha equal to 1, the attack performance begins to decrease, particularly in terms of boosted top-10. For experiments, alpha was set to 0.5.
<p align="center">
  <img src="https://github.com/Narabzad/adv-ir-empra/blob/main/interpolation_coefficient_diagram.png", width="600" height="400">
</p>

## Adversarial Examples Generated by Different Methods
We compare the generated adversarial example by EMPRA and baselines. As can be seen, the adversarial document generated by EMPRA is natural, similar to the original document, imperceptible to human and machine, and more importantly ranks as the best document by the victim black-box neural ranking model.
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
    <td class="tg-0lax">LLM (GPT-4)</td>
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
    <td class="tg-0lax">EMPRA</td>
    <td class="tg-0lax">During pregnancy, anyone can take a prenatal vitamin (folic acid, iron, and calcium) to increase their daily requirements for these nutrients. Always let your health care provider know what nutritional supplements you are taking. Prenatal vitamins consist of a variety of vitamins and minerals. During pregnancy, a womanÃ¢Â€Â™s daily intake requirements for certain nutrients, such as folic acid (folate), calcium, and iron, will increase.</td>
    <td class="tg-0lax">1</td>
  </tr>
</tbody>
</table>

## Usage
##### In order to attack a black-box neural ranking model, you can follow the process below:
1- First the top-1000 retrieved bm25 documents for the sampled queries should be re-ranked by a black-box neural ranking model using [rerank.py](https://github.com/aminbigdeli/EMPRA/blob/main//code/rerank.py). This code receives a set of sampled queries (e.g. 1000 queries) and their bm25 run file to re-rank them based on a black-box neural ranking model. For our case, we have considered the cross-encoder/ms-marco-MiniLM-L-12-v2 as the black-box model used for re-ranking the documents.
```
python rerank.py\
     -model cross-encoder/ms-marco-MiniLM-L-12-v2 \
     -collection path to MS MARCO collection \
     -queries path to queries (TSV format) \
     -run BM25 TREC run file \
     -res path to store the re-ranked run
```
2- For each query, one document from each of the 5-document segments (i.e., positions 51-60, 61-70, etc.) is selected along with the last 5 ranked documents (996-100), resulting in ten targeted documents per query using [target_doc_selector.py](https://github.com/aminbigdeli/EMPRA/blob/main//code/target_doc_selector.py)
```
python target_doc_selector.py\
     -run path to black-box model re-ranked run \
     -ouput path to output
```
3- Finally, having the ten target documents per query, we use EMPRA to perform adversarial attack using [attack.py](https://github.com/aminbigdeli/EMPRA/blob/main//code/attack.py). 
```
python attack.py\
     -collection path to collection \
     -anchor_texts path to anchor texts (TSV format) \
     -target_documents path to target documents \
     -target_documents_embeddings  #path to target documents embeddings pickle file \
     -max_num_iterations maximum number of iterations \
     -epsilon epsilon constraint of the transporter \
     -step_size step_size of the transporter \
     -output path to output folder\
```
