from enum import Enum
from typing import List

from mowl.datasets import PathDataset
from mowl.projection import projector_factory, Edge
from mowl.projection.base import ProjectionModel

from dp_cgans.embeddings import log


class ProjectionType(Enum):
    OWL2VECSTAR = "owl2vecstar"
    DL2VEC = "dl2vec"
    TAXONOMY = "taxonomy"
    TAXONOMY_WITH_RELATIONS = "taxonomy_rels"


def create_projection(projection_type: ProjectionType = ProjectionType.OWL2VECSTAR,
                      bidirectional_taxonomy: bool = False, include_literals: bool = False,
                      only_taxonomy: bool = False, use_taxonomy: bool = False, relations: List[str] = None,
                      verbose: bool = False) -> ProjectionModel:
    """
    Create a projector instance to use adjacency information of ontologies and project them into a graph.

    Args:
        projection_type: The Projection Method that is used to project the semantic information into a graph the \
            possible methods are owl2vecstar, dl2vec, taxonomy and taxonomy_rels.
        bidirectional_taxonomy: If true, then per each subClass edge one superClass edge will be generated \
            Can only be True when use_taxonomy is True.
        include_literals: If true, then the graph will also include triples involving data property assertions \
            and annotations, for owl2vecstar
        only_taxonomy: If true, then the projection will only include subClass edges, for owl2vecstar
        use_taxonomy: If true, then taxonomies will be used. Otherwise, the relations parameter should be true, \
            for taxonomy_rels
        relations: Contains a list of relations in string format, for taxonomy_rels \
            Can only be non-empty when use_taxonomy is False.
        verbose: If True, then intermediate console message will be displayed to indicate progress.

    Returns: The Projector instance

    """
    log(text=f'📽️  Creating a Projector ({projection_type.value}) for the data to be turned into a graph...',
        verbose=verbose)

    return projector_factory(method_name=projection_type.value, bidirectional_taxonomy=bidirectional_taxonomy,
                             include_literals=include_literals, only_taxonomy=only_taxonomy, taxonomy=use_taxonomy,
                             relations=relations)


def project(dataset: PathDataset, projector: ProjectionModel, verbose: bool = False) -> List[Edge]:
    """
    Use Adjacency information of Ontologies to project into a graph
    Args:
        dataset: The Ontologies
        projector: The Projection Instance
        verbose: If True, then intermediate console message will be displayed to indicate progress.

    Returns: The graph i.e. a list of edges

    """
    log(text=f'📽️  Projecting data into a graph...', verbose=verbose)
    return projector.project(dataset.ontology)