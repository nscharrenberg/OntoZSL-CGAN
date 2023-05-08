import os

import rdflib
from rdflib.query import Result

from dp_cgans.embeddings import log


def create_training_and_test_dataset(file_path: str, verbose: bool = True):
    if not os.path.exists(file_path):
        log(text=f'❌️The file {file_path} does not exists.', verbose=True)
        return

    graph = rdflib.Graph()
    graph.parse(file_path)

    print(len(graph))

    test1_query = query_get_rare_diseases_with_group(graph)
    for row in test1_query:
        print(f'{row.disease} part_of {row.group}')

    print('ENDING PART OF QUERY')

    test_query = query_get_rare_diseases_association_classes(graph)
    for row in test_query:
        print(f'{row.rd}')


def query_get_rare_diseases_with_group(graph: rdflib.Graph) -> Result:
    query = """
    SELECT ?group ?disease
    WHERE {
        ?disease_uri rdf:type owl:Class;
            <http://purl.obolibrary.org/obo/BFO_0000050> ?group .
        BIND (STR(?disease_uri) AS ?disease) .
    }
    """

    return graph.query(query)


def query_get_rare_diseases_association_classes(graph: rdflib.Graph) -> Result:
    query = """
    SELECT DISTINCT ?rd
    {
        ?subject rdfs:subClassOf :Association .
        BIND(REPLACE(STR(?subject), ".*Orpha:([0-9]+).*", "$1") AS ?sub) .
        BIND(CONCAT("http://www.orpha.net/ORDO/Orphanet_", ?sub) AS ?rd) .
    }
    """

    return graph.query(query)


def query_get_rare_diseases_by_subgroup(graph: rdflib.Graph) -> Result:
    query = """
    SELECT DISTINCT ?rd ?label
    WHERE {
        ?subject rdfs:subClassOf :Association .
        BIND(REPLACE(STR(?subject), ".*Orpha:([0-9]+).*", "$1") AS ?sub) .
        BIND(CONCAT("http://www.orpha.net/ORDO/Orphanet_", ?sub) AS ?rd) .
        BIND (URI(?rd) AS ?rd_uri) .
    
        ?rd_uri <http://purl.obolibrary.org/obo/BFO_0000050> "http://www.orpha.net/ORDO/Orphanet_207018" ;
                rdfs:label ?label .
        BIND (STR(?rd_uri) AS ?rd) .
        FILTER REGEX(?label, "[^ORPHA:0-9]") .
    }
    """

    return graph.query(query)

