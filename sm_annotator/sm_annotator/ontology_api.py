from operator import itemgetter
from typing import Dict

import requests
from elasticsearch import Elasticsearch, helpers
from fuzzywuzzy import fuzz
from rdflib import RDFS
from tqdm.auto import tqdm
from grams.kg_data.wikidatamodels import WDProperty, WDClass, QNode
from grams.inputs.linked_table import LinkedTable
from grams.algorithm.sm_wikidata import WikidataSemanticModelHelper, WDOnt


class ESStore:
    class_index = "wd-classes"
    prop_index = "wd-props"

    @staticmethod
    def search4class(eshost: str, query: str):
        if query.isdigit():
            query = "Q" + query
        resp = requests.post(f"{eshost}/{ESStore.class_index}/_search", json={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        'id^10',
                        'label^5',
                        'aliases^3',
                        'description'
                    ]
                }
            }
        })
        assert resp.status_code == 200, resp.status_code
        data = resp.json()
        return [(r['_source'], r['_score'])
                for r in data['hits']['hits']]

    @staticmethod
    def search2prop(eshost: str, query: str):
        if query.isdigit():
            query = "P" + query
        resp = requests.post(f"{eshost}/{ESStore.prop_index}/_search", json={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        'id^10',
                        'label^5',
                        'aliases^3',
                        'description'
                    ]
                }
            }
        })
        data = resp.json()
        return [(r['_source'], r['_score'])
                for r in data['hits']['hits']]

    @staticmethod
    def load2store(eshost):
        ESStore.create_index(eshost)

        props = WDProperty.from_file()
        classes = WDClass.from_file()
        assert len(set(props.keys()).intersection(classes.keys())) == 0

        es = Elasticsearch([eshost])

        docs = []
        for c in props.values():
            doc = {
                k: getattr(c, k)
                for k in ['id', 'label', 'aliases', 'description']
            }
            doc['_id'] = c.id
            docs.append(doc)

        for i in tqdm(range(0, len(docs), 128)):
            helpers.bulk(es, docs[i:i + 128], index=ESStore.prop_index)
        helpers.bulk(es, [
            {
                "_id": str(RDFS.label),
                "id": str(RDFS.label),
                "label": "rdfs:label",
                "aliases": ["label"],
                "description": ["label of a resource"]
            }
        ], index=ESStore.prop_index)

        docs = []
        for c in classes.values():
            doc = {
                k: getattr(c, k)
                for k in ['id', 'label', 'aliases', 'description']
            }
            doc['_id'] = c.id
            docs.append(doc)
        for i in tqdm(range(0, len(docs), 128)):
            helpers.bulk(es, docs[i:i + 128], index=ESStore.class_index)
        helpers.bulk(es, [
            {
                "_id": "http://wikiba.se/ontology#Statement",
                "id": "http://wikiba.se/ontology#Statement",
                "label": "wikibase:Statement",
                "aliases": ["statement"],
                "description": "Wikidata Statement"
            }
        ], index=ESStore.class_index)

    @staticmethod
    def create_index(eshost):
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "keyword_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase"]
                        },
                        "default": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "2_16_edgegrams"]
                        },
                        "default_search": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "16_truncate"]
                        },
                    },
                    "filter": {
                        "2_16_edgegrams": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 16
                        },
                        "16_truncate": {
                            "type": "truncate",
                            "length": 16
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {
                        "type": "text",
                        "analyzer": "keyword_analyzer",
                    }
                }
            }
        }

        for index in [ESStore.class_index, ESStore.prop_index]:
            resp = requests.put(f"{eshost}/{index}", json=index_settings)
            assert resp.status_code == 200, resp.text


class OntologyAPI:
    def __init__(self, eshost: str, ontclasses: Dict[str, WDClass],
                 ontprops: Dict[str, WDProperty]):
        self.eshost = eshost
        self.ontclasses = ontclasses
        self.ontprops = ontprops

    def search_class(self, query: str):
        resp = []
        for node, score in ESStore.search4class(self.eshost, query)[:20]:
            resp.append({
                "uri": WDOnt.get_qnode_uri(node['id']) if node['id'].startswith("Q") else node['id'],
                "label": f"{node['label']} ({node['id']})" if node['id'].startswith("Q") else node['label'],
                "description": node['description'],
                "score": score
            })
        return resp

    def search_predicate(self, query: str):
        resp = []
        for node, score in ESStore.search2prop(self.eshost, query)[:20]:
            resp.append({
                "uri": WDOnt.get_prop_uri(node['id']) if node['id'].startswith("P") else node['id'],
                "label": f"{node['label']} ({node['id']})" if node['id'].startswith("P") else node['label'],
                "description": node['description'],
                "score": score
            })
        return resp


if __name__ == '__main__':
    eshost = "http://mira.isi.edu:9200"

    assert requests.delete(f"{eshost}/{ESStore.prop_index}").status_code == 200 or True
    assert requests.delete(f"{eshost}/{ESStore.class_index}").status_code == 200 or True
    ESStore.load2store(eshost)
    exit(0)
    query = 'adm terr entit'
    query = 'q56061'
    resp = ESStore.search4class("http://mira.isi.edu:9200", query)
    # resp = ESStore.update_index("http://mira.isi.edu:9200")
    # resp = OntologyAPI("http://localhost:9200", {}, {}).search_class("Statement")

    print(resp)