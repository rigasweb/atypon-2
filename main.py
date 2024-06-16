from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch


es = Elasticsearch(['http://localhost:9200']) 
index_name = "scientific_articles"

app = FastAPI()

class QueryRequest(BaseModel):
    query_title: str
    query_abstract: str
    top_n: int = 10

# Load pre-trained transformer model and tokenizer
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def query_elasticsearch(query_title, query_abstract, top_n=10):
    query_text = query_title + ' ' + query_abstract
    query_embedding = generate_embedding(query_text)
    
    # Lexical search
    lexical_query = {
        "bool": {
            "should": [
                {"match": {"title": query_title}},
                {"match": {"abstract": query_abstract}}
            ]
        }
    }
    
    # Semantic search using script score for vector similarity
    semantic_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding.tolist()}
            }
        }
    }
    
    # Execute both queries
    lexical_results = es.search(index=index_name, body={"query": lexical_query, "size": top_n})
    semantic_results = es.search(index=index_name, body={"query": semantic_query, "size": top_n})
    
    # Combine and rerank results
    results = {hit['_id']: hit['_score'] for hit in lexical_results['hits']['hits']}
    for hit in semantic_results['hits']['hits']:
        if hit['_id'] in results:
            results[hit['_id']] += hit['_score']
        else:
            results[hit['_id']] = hit['_score']
    
    # Get top N results
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [res for res in sorted_results]

@app.post("/query_elasticsearch")
async def search_es(query_request: QueryRequest):
    try:
        top_n_results = query_elasticsearch(query_request.query_title, query_request.query_abstract, query_request.top_n)
        return {"top_n_results": top_n_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)