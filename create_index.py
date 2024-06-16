from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm 
import json 
import numpy as np

es = Elasticsearch(['http://localhost:9200'])

# Define the Elasticsearch index settings and mappings
index_name = "scientific_articles"

index_config = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard"
                }
            }
        }
    },

"mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "title": {"type": "text"},
      "authors": {"type": "nested",
                "properties": {
                "name": {"type": "text"},
                "id": {"type": "keyword"}}},
      "venue": {"type": "object",
                "properties": {
                "raw": {"type": "text"}}},
      "year": {"type": "integer"},
      "n_citation": {"type": "integer"},
      "page_start": {"type": "keyword"},
      "page_end": {"type": "keyword"},
      "doc_type": {"type": "keyword"},
      "publisher": {"type": "text"},
      "volume": {"type": "keyword"},
      "issue": {"type": "keyword"},
      "fos": {"type": "nested",
                "properties": {
                "name": {"type": "text"},
                "w": {"type": "integer"}}},
      "abstract": {"type": "text"},
      "embedding": {"type": "dense_vector", "dims": 768}
    }
}}

# Load pre-trained transformer model and tokenizer
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def generate_embedding(text: str) -> np.array:
    """
    Given a text generate the embeddings

    :param text: the string to generate the embeddings
    :return: the embeddings as a numpy array
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def index_documents(file_path: str) -> None:
    """
    Add documents to an elastic search index using the bulk method

    :param file_path: the path to the file that contains the documents
    :return: None
    """
    with open(file_path, 'r') as f:
        documents = [json.loads(line) for line in f]
    
    actions = []
    for i,doc in tqdm(enumerate(documents)):
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        text = title + ' ' + abstract
        embedding = generate_embedding(text)
        
        action = {
                    "_index": "scientific_articles",
                    "_source": {
                        "id": doc.get("id"),
                        "title": title,
                        "authors": doc.get("authors"),
                        "venue": doc.get("venue"),
                        "year": doc.get("year"),
                        "n_citation": doc.get("n_citation"),
                        "page_start": doc.get("page_start"),
                        "page_end": doc.get("page_end"),
                        "doc_type": doc.get("doc_type"),
                        "publisher": doc.get("publisher"),
                        "volume": doc.get("volume"),
                        "issue": doc.get("issue"),
                        "fos": doc.get("fos"),
                        "abstract": abstract,
                        "embedding": embedding
                    }
            }
        actions.append(action)

        # every 100 documents add them to es 
        if i % 100 == 0 :
            bulk(es, actions)
            actions = []


if __name__ == "__main__":

    # if index doesn't exist, create it 
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_config)

    # add documents
    index_documents('dblpv11_sample.json')