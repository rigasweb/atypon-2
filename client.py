import requests

url = "http://localhost:8000/query_elasticsearch"  

# Example query parameters
query_title = "A study on machine learning"
query_abstract = "This paper explores various machine learning techniques and their applications."
top_n = 10

payload = {
    "query_title": query_title,
    "query_abstract": query_abstract,
    "top_n": top_n
}


if __name__ == "__main__":
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Response JSON:")
        print(response.json()) 
    else:
        print(f"Error: HTTP status code {response.status_code}")
        print(response.text)  