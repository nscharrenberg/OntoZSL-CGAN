from enum import Enum
from typing import List

from mowl.datasets import PathDataset
from mowl.projection import projector_factory, Edge
from mowl.projection.base import ProjectionModel

from dp_cgans.utils.logging import log


class ProjectionType(Enum):
    """Enumeration of supported projection types."""
    OWL2VECSTAR = "owl2vecstar"
    DL2VEC = "dl2vec"
    TAXONOMY = "taxonomy"
    TAXONOMY_WITH_RELATIONS = "taxonomy_rels"


def create_projection(projection_type: ProjectionType = ProjectionType.OWL2VECSTAR,
                      bidirectional_taxonomy: bool = False, include_literals: bool = False,
                      only_taxonomy: bool = False, use_taxonomy: bool = False, relations: List[str] = None,
                      verbose: bool = False) -> ProjectionModel:
    """
    Create a projection model based on the specified parameters.

    Args:
        projection_type (ProjectionType, optional): The type of projection to create.
            Defaults to ProjectionType.OWL2VECSTAR.
        bidirectional_taxonomy (bool, optional): Whether the taxonomy should be bidirectional.
            Defaults to False.
        include_literals (bool, optional): Whether literals should be included in the projection.
            Defaults to False.
        only_taxonomy (bool, optional): Whether only the taxonomy should be included in the projection.
            Defaults to False.
        use_taxonomy (bool, optional): Whether the taxonomy should be used in the projection.
            Defaults to False.
        relations (List[str], optional): Additional relations to be included in the projection.
            Defaults to None.
        verbose (bool, optional): Whether verbose logging should be enabled.
            Defaults to False.

    Returns:
        ProjectionModel: The created projection model.
    """
    log(text=f'📽️  Creating a Projector ({projection_type.value}) for the data to be turned into a graph...',
        verbose=verbose)

    return projector_factory(method_name=projection_type.value, bidirectional_taxonomy=bidirectional_taxonomy,
                             include_literals=include_literals, only_taxonomy=only_taxonomy, taxonomy=use_taxonomy,
                             relations=relations)


def project(dataset: PathDataset, projector: ProjectionModel, verbose: bool = False) -> List[Edge]:
    """
    Project data from a PathDataset object into a graph using a projection model.

    Args:
        dataset (PathDataset): The dataset to project.
        projector (ProjectionModel): The projection model to use.
        verbose (bool, optional): Whether verbose logging should be enabled.
            Defaults to False.

    Returns:
        List[Edge]: The projected edges.
    """
    log(text=f'📽️  Projecting data into a graph...', verbose=verbose)
    return projector.project(dataset.ontology)