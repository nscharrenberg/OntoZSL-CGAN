import random

import rdflib
from rdflib import URIRef
from rdflib.query import Result

from dp_cgans.utils.logging import log, LogLevel


def populate_rd_sets(rd_set: set, rd_list: list, rd_groups_dict: dict, method: str = "no_groups", shuffle: bool = False,
                     unseen_percentage: float = 0.2, selected_seen_rds: list = [], selected_unseen_rds: list = [],
                     verbose: bool = True):
    """
    Populates two sets, 'seen_rds_set' and 'unseen_rds_set', based on the given parameters and the specified method.
    It creates a balanced distribution of seen and unseen rare diseases.

    Parameters:
        rd_set (set): A set of rare diseases.
        rd_list (list): A list of rare diseases.
        rd_groups_dict (dict): A dictionary mapping rare disease groups to the corresponding rare diseases.
        method (str): The method to determine the distribution of seen and unseen rare diseases. Default is "no_groups".
        shuffle (bool): A flag indicating whether to shuffle the order of rare diseases before populating the sets.
        unseen_percentage (float): The desired percentage of unseen rare diseases in the total set.
        selected_seen_rds (list): A list of rare diseases to be included in the seen set.
        selected_unseen_rds (list): A list of rare diseases to be included in the unseen set.
        verbose (bool): A flag indicating whether to log detailed information during the population process.

    Returns:
        tuple: A tuple containing two sets: 'seen_rds_set' and 'unseen_rds_set'.
    """
    log(text=f"Initializing Populating Rare Disease Sets...", verbose=verbose)
    seen_rds_set = set()
    unseen_rds_set = set()
    log(text=f"Establishing selected method", verbose=verbose)

    methods = {
        "no_groups",
        "parts_groups",
        "whole_groups",
        "selected_rds"
    }

    if method not in methods:
        log(text=f"Method {method} is not an option from {methods}", level=LogLevel.ERROR)

    log(text=f"The following method will be performed: {method}.", verbose=verbose)

    total_rds = len(rd_set)

    if method == "no_groups":
        indexes = [i for i in range(len(rd_list))]

        if shuffle:
            log(text=f"Shuffling...", verbose=verbose)
            random.shuffle(indexes)

        log(text=f"Creating seen and unseen Rare Disease sets...", verbose=verbose)

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
            log(text=f"Shuffling...", verbose=verbose)
            random.shuffle(keys)

        log(text=f"Creating seen and unseen Rare Disease sets...", verbose=verbose)
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
            log(text=f"Shuffling...", verbose=verbose)
            random.shuffle(keys)

        log(text=f"Creating seen and unseen Rare Disease sets...", verbose=verbose)
        for group in keys:
            rds = rd_groups_dict[group]

            if len(unseen_rds_set) > total_rds * unseen_percentage:
                for rd in rds:
                    if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                        seen_rds_set.add(rd)

    elif method == "selected_rds":
        log(text=f"Creating seen and unseen Rare Disease sets...", verbose=verbose)
        for rd in selected_unseen_rds:
            if rd in rd_set:
                if rd not in unseen_rds_set:
                    unseen_rds_set.add(rd)
            else:
                log(text=f"The Rare Disease \"{rd}\" is unknown.", level=LogLevel.ERROR)

        log(text=f"Creating seen and unseen Rare Disease sets...", verbose=verbose)
        if len(selected_seen_rds) > 0:
            for rd in selected_seen_rds:
                if rd in rd_set:
                    if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                        seen_rds_set.add(rd)
                else:
                    log(text=f"The Rare Disease \"{rd}\" is unknown.", level=LogLevel.ERROR)
        else:
            for rds in rd_groups_dict.values():
                for rd in rds:
                    if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                        seen_rds_set.add(rd)

    if method not in ["no_groups", "selected_rds"]:
        log(text=f"method indicated grouping of diseases is taking place. Therefore, an additional step is performed to add them to the seen Rare Disease Sets...",
            verbose=verbose)
        # TODO: allow some of these to be added to unseen_rds_set?
        # adding the RDs that aren't part of a RD group, in the seen set
        for rd in rd_list:
            if (rd not in seen_rds_set) and (rd not in unseen_rds_set):
                seen_rds_set.add(rd)

    log(text=f"The number of seen Rare Diseases is {len(seen_rds_set)} ({(len(seen_rds_set) / len(rd_set) * 100):.2f}%)",
        level=LogLevel.INFO, verbose=verbose)
    log(text=f"The number of unseen Rare Diseases is {len(unseen_rds_set)} ({(len(unseen_rds_set) / len(rd_set) * 100):.2f}%)",
        level=LogLevel.INFO, verbose=verbose)
    log(text=f"The seen and unseen Rare Disease Sets have been created.", level=LogLevel.OK, verbose=verbose)

    return seen_rds_set, unseen_rds_set


def build_dictionary(graph: rdflib.Graph, verbose: bool = True):
    """
    Builds a dictionary of rare disease groups and their associated rare diseases based on the given RDF graph.
    It retrieves rare diseases and their group information from the graph and constructs the dictionary.

    Parameters:
        graph (rdflib.Graph): An RDF graph containing rare disease data.
        verbose (bool): A flag indicating whether to log detailed information during the dictionary building process.

    Returns:
        tuple: A tuple containing the following:
            - rd_groups_dict (dict): A dictionary mapping rare disease groups to the corresponding rare diseases.
            - rd_set (set): A set of all unique rare diseases.
            - rd_list (list): A list of all rare diseases.
    """
    rd_list = []
    rd_groups_dict = {}

    log(text=f"Retrieving Rare Disease Association Classes", verbose=verbose)
    rds_associated_classes_query = query_get_rare_diseases_association_classes(graph)

    log(text=f"Assigning Association Classes to list of rare disease", verbose=verbose)
    for row in rds_associated_classes_query:
        rd_list.append(str(row.rd))

    rd_set = set(rd_list)

    log(text=f"Retrieving Rare Disease including their group", verbose=verbose)
    rds_corresponding_groups_query = query_get_rare_diseases_with_group(graph)

    log(text=f"Building Dictionary of Groups and their Association Rare Diseases", verbose=verbose)
    for row in rds_corresponding_groups_query:
        rd = str(row.rd)
        group = str(row.group)

        if rd in rd_set:
            if group not in rd_groups_dict:
                rd_groups_dict[group] = []

            if rd not in rd_groups_dict[group]:
                rd_groups_dict[group].append(rd)

    log(text=f"The number of unique, seen in Associated class Rare Diseases are {len(rd_list)}, {len(rd_set)}",
        level=LogLevel.INFO)
    log(text=f"The number of Rare Disease Groups is {len(rd_groups_dict)}", level=LogLevel.INFO)

    len_sum = sum(len(dct) for dct in rd_groups_dict.values())

    log(text=f"The total number of rare diseases in the groups are {len_sum}", level=LogLevel.INFO, verbose=verbose)
    log(text=f"The Average amount of Rare Diseases per group is {len_sum / len(rd_groups_dict)}", level=LogLevel.INFO,
        verbose=verbose)
    log(text=f"The dictionary has been successfully build.", level=LogLevel.OK)

    return rd_groups_dict, rd_set, rd_list


def query_get_rare_diseases_with_group(graph: rdflib.Graph) -> Result:
    """
    Queries the given RDF graph to retrieve rare diseases and their associated groups.

    Parameters:
        graph (rdflib.Graph): An RDF graph containing rare disease data.

    Returns:
        rdflib.plugins.sparql.Result: A 'Result' object containing the query results.
    """
    return graph.query("""
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
        """)


def query_get_rare_diseases_association_classes(graph: rdflib.Graph) -> Result:
    """
    Queries the given RDF graph to retrieve rare diseases and their association classes.

    Parameters:
        graph (rdflib.Graph): An RDF graph containing rare disease data.

    Returns:
        rdflib.plugins.sparql.Result: A 'Result' object containing the query results.
    """
    return graph.query("""
    SELECT DISTINCT ?rd
    {
        ?subject rdfs:subClassOf HOOM:Association .
        BIND(REPLACE(STR(?subject), ".*Orpha:([0-9]+).*", "$1") AS ?sub) .
        BIND(CONCAT("http://www.orpha.net/ORDO/Orphanet_", ?sub) AS ?rd) .
    }
    """)


def query_get_rare_diseases_by_subgroup(graph: rdflib.Graph, uri: str) -> Result:
    """
    Queries the given RDF graph to retrieve rare diseases belonging to a specific subgroup.

    Parameters:
        graph (rdflib.Graph): An RDF graph containing rare disease data.
        uri (str): The URI of the subgroup.

    Returns:
        rdflib.plugins.sparql.Result: A 'Result' object containing the query results.
    """
    return graph.query("""
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
    """, initBindings={'given_uri': URIRef(uri)})


def query_get_rare_diseases_by_word(graph: rdflib.Graph, word: str) -> Result:
    """
    Queries the given RDF graph to retrieve rare diseases containing a specific word in their labels.

    Parameters:
        graph (rdflib.Graph): An RDF graph containing rare disease data.
        word (str): The word to search for in rare disease labels.

    Returns:
        rdflib.plugins.sparql.Result: A 'Result' object containing the query results.

    Note:
        - TODO: Update this function to use 'initBindings' instead of potentially unsafe string escaping, issue with using 'Literal(word)' directly is that it may result in an empty query result.
    """
    return graph.query("""
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
    """)
