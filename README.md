# Querying Elastic Search

## Set up Elastic Search and Kibana:
$docker network create elastic
$docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.22
$docker run --name es01-test --net elastic -p 127.0.0.1:9200:9200 -p 127.0.0.1:9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.17.22

## in a different terminal 
$docker pull docker.elastic.co/kibana/kibana:7.17.22
$docker run --name kib01-test --net elastic -p 127.0.0.1:5601:5601 -e "ELASTICSEARCH_HOSTS=http://es01-test:9200" docker.elastic.co/kibana/kibana:7.17.22

## Create index and launch the app
$python create_index.py
$python app.py

## Send a request:
$python client.py
