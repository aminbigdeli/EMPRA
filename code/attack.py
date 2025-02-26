import vec2text
import torch
import pandas as pd
import torch
import torch.nn.functional as F
import re
from tqdm import tqdm
import pickle
import torch
from openai import OpenAI
import argparse
import time

client = OpenAI(
  api_key='',  # OPEN AI API KEY
)

def get_embedding(text, model="text-embedding-ada-002"):
    embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
    embedding = torch.tensor([embedding])
    return embedding

def invert_embeddings(corrector, embeddings):
    text = vec2text.invert_embeddings(
        embeddings=embeddings.cuda(),
        corrector=corrector,
        num_steps = 1
    )[0]
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    test = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.strip()
    return text

def attack(corrector, anchor_text, sentence, num_iterations, epsilon, step_size):
    sentence_embedding = get_embedding(sentence)
    anchor_embedding = get_embedding(anchor_text)
    perturbed_embedding = torch.clone(sentence_embedding).requires_grad_(True)
    for iteration in range(num_iterations):
        closeness_similarity = F.cosine_similarity(perturbed_embedding, anchor_embedding, dim=1).mean()
        gradient = torch.autograd.grad(closeness_similarity, perturbed_embedding)[0]
        perturbations = torch.clamp(gradient, -epsilon, epsilon)
        perturbed_embedding.data = perturbed_embedding + step_size * perturbations
        perturbed_embedding.data = F.normalize(perturbed_embedding, p=2, dim=1)
        purturbed_sentence_text = invert_embeddings(corrector, perturbed_embedding)
    purturbed_sentence_text = invert_embeddings(corrector, perturbed_embedding)
    return purturbed_sentence_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-anchor_texts', type=str) #path to anchor texts
    parser.add_argument('-collection', type=str) #path to collection (MS MARCO collection)
    parser.add_argument('-target_documents', type=str) #path to target documents
    parser.add_argument('-target_documents_embeddings', type=str) #path to target documents embeddings pickle file
    parser.add_argument('-max_num_iterations', type=int, default=10) #max number of iterations
    parser.add_argument('-epsilon', type=float, default=0.01) #epsilon
    parser.add_argument('-step_size', type=float, default=0.1) #step size
    parser.add_argument('-output', type=int) #path to output directory
    args = parser.parse_args()

    start_time = time.time()
    corrector = vec2text.load_corrector("text-embedding-ada-002")
    anchor_texts_df = pd.read_csv(args.anchor_texts, sep = "\t", names = ['qid', 'text'])
    collection = pd.read_csv(args.collection, sep = "\t", names = ['pid', 'passage'])
    corpus = {}
    for row in collection.values.tolist():
        corpus[row[0]] = row[1]
    target_documents = pd.read_csv(args.target_documents, sep = " ", names =['qid', 'pid', 'rank', 'score'])
    with open(args.target_documents_embeddings, 'rb') as handle:
      target_documents_embeddings = pickle.load(handle)

    grps = target_documents.groupby(['qid'])
    result = {}
    for name, group in tqdm(grps):
        qid = name[0]
        anchor_text = anchor_texts_df[anchor_texts_df['qid'] == qid]['text'].values.tolist()[0]
        target_documents_info = []
        for index, row in group.iterrows():
            doc_id = row['pid']
            rank = row['rank']
            initial_score = row['score']
            target_documents_info.append([doc_id, rank, initial_score])
        target_documents_dict = {}
        for target_document in target_documents_info:
            target_document_id = int(target_documents_info[0])
            document_info = target_documents_embeddings[target_document_id]
            sentences = document_info['sentences']
            sentence_embeddings = document_info['sentence_embeddings']
            perturbed_sentences = []
            for i in range(len(sentences)):
                perturbed_sentence = attack(anchor_text, sentence_embeddings[i], max_num_iterations, epsilon, step_size)
                perturbed_sentences.append(perturbed_sentence)
            document_info['perturbed_sentences'] = perturbed_sentences
            result[target_document_id] = document_info

        result[qid] = target_documents_dict

        with open(args.output + "/attack_output.pickle", "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("--- Total Execution Time: %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
