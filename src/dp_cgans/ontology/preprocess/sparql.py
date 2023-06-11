import random

import rdflib
from rdflib import URIRef
from rdflib.query import Result

from dp_cgans.utils.logging import log


def populate_rd_sets(rd_set: set, rd_list: list, rd_groups_dict: dict, method: str = "no_groups", shuffle: bool = False,
                     unseen_percentage: float = 0.2, selected_seen_rds: list = [], selected_unseen_rds: list = [],
                     verbose: bool = True):
    log(f"📖 Progress: Initializing Populating Rare Disease Sets...", verbose=verbose)

    seen_rds_set = set()
    unseen_rds_set = set()

    log(f"📖 Progress: Establishing selected method", verbose=verbose)
    methods = {
        "no_groups",
        "parts_groups",
        "whole_groups",
        "selected_rds"
    }

    if method not in methods:
        raise Exception(f"Method {method} is not an option from {methods}")

    log(f"📖 Progress: The following method will be performed: {method}", verbose=verbose)
    total_rds = len(rd_set)

    if method == "no_groups":
        indexes = [i for i in range(len(rd_list))]

        if shuffle:
            log(f"📖 Progress: Shuffling...", verbose=verbose)
            random.shuffle(indexes)

        log(f"📖 Progress: Creating seen and unseen Rare Disease Sets...", verbose=verbose)
        for ind in indexes:
            rd = rd_list[ind]

            if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                if len(unseen_rds_set) > total_rds * unseen_percentage:
                    seen_rds_set.add(rd)
                else:
                    unseen_rds_set.add(rd)

    elif method == "parts_groups":
        keys = list(rd_groups_dict.keys())

        if shuffle:
            log(f"📖 Progress: Shuffling...", verbose=verbose)
            random.shuffle(keys)

        log(f"📖 Progress: Creating seen and unseen Rare Disease Sets...", verbose=verbose)
        for group in keys:
            rds = rd_groups_dict[group]

            unseen_count = 0

            if len(unseen_rds_set) > total_rds * unseen_percentage:
                unseen_count = len(unseen_rds_set)

            for rd in rds:
                if rd in unseen_rds_set:
                    unseen_count += 1
                else:
                    unseen_rds_set.add(rd)
                    unseen_count += 1

    elif method == "whole_groups":
        keys = list(rd_groups_dict.keys())

        if shuffle:
            log(f"📖 Progress: Shuffling...", verbose=verbose)
            random.shuffle(keys)

        log(f"📖 Progress: Creating seen and unseen Rare Disease Sets...", verbose=verbose)
        for group in keys:
            rds = rd_groups_dict[group]

            if len(unseen_rds_set) > total_rds * unseen_percentage:
                for rd in rds:
                    if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                        seen_rds_set.add(rd)

    elif method == "selected_rds":
        log(f"📖 Progress: Creating unseen Rare Disease Sets...", verbose=verbose)
        for rd in selected_unseen_rds:
            if rd in rd_set:
                if rd not in unseen_rds_set:
                    unseen_rds_set.add(rd)
            else:
                raise Exception(f"The Rare Disease {rd} is unknown")

        log(f"📖 Progress: Creating seen Rare Disease Sets...", verbose=verbose)
        if len(selected_seen_rds) > 0:
            for rd in selected_seen_rds:
                if rd in rd_set:
                    if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                        seen_rds_set.add(rd)
                else:
                    raise Exception(f"The Rare Disease {rd} is unknown")
        else:
            for rds in rd_groups_dict.values():
                for rd in rds:
                    if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                        seen_rds_set.add(rd)

    if method not in ["no_groups", "selected_rds"]:
        log(f"📖 Progress: method indicated grouping of diseases it taking place. Therefore, an additional step is performed to add them to the seen Rare Disease Sets...",
            verbose=verbose)
        # TODO: allow some of these to be added to unseen_rds_set?
        # adding the RDs that aren't part of a RD group, in the seen set
        for rd in rd_list:
            if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                seen_rds_set.add(rd)

    log(f"📖 Info: The number of seen Rare Diseases is {len(seen_rds_set)} ({(len(seen_rds_set) / len(rd_set) * 100):.2f}%)",
        verbose=verbose)
    log(f"📖 Info: The number of unseen Rare Diseases is {len(unseen_rds_set)} ({(len(unseen_rds_set) / len(rd_set) * 100):.2f}%)",
        verbose=verbose)

    log(f"✅ Success! The seen and unseen Rare Disease Sets have been created.",
        verbose=verbose)

    return seen_rds_set, unseen_rds_set


def build_dictionary(graph: rdflib.Graph, verbose: bool = True):
    rd_list = []
    rd_groups_dict = {}

    log("📖 Querying: Retrieving Rare Disease Association Classes", verbose=verbose)
    rds_associated_classes_query = query_get_rare_diseases_association_classes(graph)

    log("📖 Processing: Assigning Association Classes to list of rare diseases", verbose=verbose)
    for row in rds_associated_classes_query:
        rd_list.append(str(row.rd))

    rd_set = set(rd_list)

    log("📖 Querying: Retrieving Rare Disease including their group", verbose=verbose)
    rds_corresponding_groups_query = query_get_rare_diseases_with_group(graph)

    log("📖 Processing: building Dictionary of Groups and their Association Rare Diseases", verbose=verbose)
    for row in rds_corresponding_groups_query:
        rd = str(row.rd)
        group = str(row.group)

        if rd in rd_set:
            if group not in rd_groups_dict:
                rd_groups_dict[group] = []

            if rd not in rd_groups_dict[group]:
                rd_groups_dict[group].append(rd)

    log(f"📖 Info: The number of unique, seen in Associated class Rare Diseases are {len(rd_list)}, {len(rd_set)}",
        verbose=verbose)
    log(f"📖 Info: The number of Rare Disease Groups is {len(rd_groups_dict)}", verbose=verbose)
    len_sum = sum(len(dct) for dct in rd_groups_dict.values())

    log(f"📖 Info: The total number of rare diseases in the groups are {len_sum}",
        verbose=verbose)
    log(f"📖 Info: The Average amount of Rare Diseases per group is {len_sum / len(rd_groups_dict)}",
        verbose=verbose)

    log(f"✅ Success! The dictionary has been successfully build",
        verbose=verbose)
    return rd_groups_dict, rd_set, rd_list


def query_get_rare_diseases_with_group(graph: rdflib.Graph) -> Result:
    query = """
    SELECT ?rd ?group
    WHERE {
      ?rd a owl:Class .
      ?group a owl:Class .
      ?rd rdfs:subClassOf [ 
        a owl:Restriction ;
        owl:onProperty <http://purl.obolibrary.org/obo/BFO_0000050> ;
        owl:someValuesFrom ?group
      ] .
    }
    """

    return graph.query(query)


def query_get_rare_diseases_association_classes(graph: rdflib.Graph) -> Result:
    query = """
    SELECT DISTINCT ?rd
    {
        ?subject rdfs:subClassOf HOOM:Association .
        BIND(REPLACE(STR(?subject), ".*Orpha:([0-9]+).*", "$1") AS ?sub) .
        BIND(CONCAT("http://www.orpha.net/ORDO/Orphanet_", ?sub) AS ?rd) .
    }
    """

    return graph.query(query)


def query_get_rare_diseases_by_subgroup(graph: rdflib.Graph, uri: str) -> Result:
    query = """
    SELECT DISTINCT ?rd ?label
    {
        ?subject rdfs:subClassOf HOOM:Association .
        BIND(REPLACE(STR(?subject), ".*Orpha:([0-9]+).*", "$1") AS ?sub) .
        BIND(CONCAT("http://www.orpha.net/ORDO/Orphanet_", ?sub) AS ?rd_1) .
  		BIND (URI(?rd_1) AS ?rd_uri) .

        ?rd_uri rdfs:subClassOf [ 
          a owl:Restriction ;
          owl:onProperty <http://purl.obolibrary.org/obo/BFO_0000050> ;
          owl:someValuesFrom ?given_uri
        ] .
  		OPTIONAL { ?rd_uri rdfs:label ?label}
  		BIND (STR(?rd_uri) AS ?rd) .
  		FILTER REGEX(?label, "[^ORPHA:0-9]") .
    }
    """

    return graph.query(query, initBindings={'given_uri': URIRef(uri)})


def query_get_rare_diseases_by_word(graph: rdflib.Graph, word: str) -> Result:
    """
    Retrieves the Association classes that contain a specific given word.

    TODO: use `initBindings` instead of a potentially `unsafe` escape. \
        The issue with this is that `Literal(word)` would give an empty result, \
        and none of the other aspects seem to work either.

    Args:
        graph: the Graph to read from
        word: the word to filter by

    Returns: the results of the query

    """

    query = """
    SELECT DISTINCT ?rd ?label
    {
        ?subject rdfs:subClassOf HOOM:Association .
        BIND(REPLACE(STR(?subject), ".*Orpha:([0-9]+).*", "$1") AS ?sub) .
        BIND(CONCAT("http://www.orpha.net/ORDO/Orphanet_", ?sub) AS ?rd_1) .
  		BIND (URI(?rd_1) AS ?rd_uri) .

  		OPTIONAL { ?rd_uri rdfs:label ?label}
  		BIND (STR(?rd_uri) AS ?rd) .
  		FILTER REGEX(?label, "[^ORPHA:0-9]") .
  		FILTER CONTAINS(?label, \"""" + word + """\") .
    }
    """

    return graph.query(query)