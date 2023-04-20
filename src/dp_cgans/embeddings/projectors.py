from enum import Enum

from mowl.datasets import PathDataset
from mowl.projection import projector_factory, Edge
from typing import List

from mowl.projection.base import ProjectionModel

from src.dp_cgans.embeddings.utils import log


class ProjectionType(Enum):
    OWL2VECSTAR = "owl2vecstar"
    DL2VEC = "dl2vec"
    TAXONOMY = "taxonomy"
    TAXONOMY_WITH_RELATIONS = "taxonomy_rels"


def create_projection(projection_type: ProjectionType = ProjectionType.OWL2VECSTAR,
                      bidirectional_taxonomy: bool = False,
                      include_literals: bool = False, only_taxonomy: bool = False, use_taxonomy: bool = False,
                      relations: List[str] = None, verbose: bool = False) -> ProjectionModel:
    log(text=f'📽️  Creating a Projector ({projection_type.value}) for the data to be turned into a graph...',
        verbose=verbose)
    return projector_factory(method_name=projection_type.value, bidirectional_taxonomy=bidirectional_taxonomy,
                             include_literals=include_literals, only_taxonomy=only_taxonomy, taxonomy=use_taxonomy,
                             relations=relations)


def project(dataset: PathDataset, projector: ProjectionModel, verbose=False) -> List[Edge]:
    log(text=f'📽️  Projecting data into a graph...',
        verbose=verbose)
    return projector.project(dataset.ontology)
