from enum import Enum
from typing import List

from gensim.models.word2vec import LineSentence
from mowl.datasets import PathDataset
from mowl.kge import KGEModel
from mowl.projection import Edge
from mowl.walking import walker_factory
from mowl.walking.walking import WalkingModel
from pykeen.models import TransE

from dp_cgans.utils.logging import log


class WalkerType(Enum):
    DEEPWALK = "deepwalk"
    NODE2VEC = "node2vec"


def create_random_walker(walker_type: WalkerType = WalkerType.DEEPWALK, num_walks: int = 10, walk_length: int = 10,
                         outfile: any = "tmp/walks.txt", workers: int = 1, alpha: float = 0., p: float = 1.,
                         q: float = 1.,
                         verbose: bool = False) -> WalkingModel:
    """
    Create a Random Walker instance to walk the edges of the graph to create "sentences"

    Args:
        walker_type: The walker Method that is used to learn the representations for vertices in the projected graph \
            the possible methods are deepwalk and node2vec
        num_walks: The number of walks
        walk_length: The length of a walk
        outfile: The output File of the Walk
        workers: The amount of workers
        alpha: The probability of a restart, for DeepWalk
        p: The Return Hyperparameter, for Node2Vec
        q: The In-Out Hyperparameter, for Node2Vec
        verbose: If True, then intermediate console message will be displayed to indicate progress.

    Returns: The Random Walker instance

    """
    log(text=f'🧍️  Creating a Walker ({walker_type.value}) to be used for the graph to be walked on...',
        verbose=verbose)

    return walker_factory(method_name=walker_type.value, num_walks=num_walks, walk_length=walk_length, outfile=outfile,
                          workers=workers, alpha=alpha, p=p, q=q)


def pykeen_transe(dataset: PathDataset, dimension: int = 50, epochs: int = 10, batch_size: int = 32, seed: int = 42):
    triples_factory = Edge.as_pykeen(dataset.ontology, create_inverse_triples=True)
    pk_model = TransE(triples_factory=triples_factory, embedding_dim=dimension, random_seed=seed)
    model = KGEModel(triples_factory, pk_model, epochs=epochs, batch_size=batch_size)
    model.train()

    return model


def walk(edges: List[Edge], walker: WalkingModel, verbose: bool = False) -> LineSentence:
    """
    Walks the edges of the graph using the created Random Walker Instance

    Args:
        edges: The edges of the graph
        walker: The Random Walker Instance to use
        verbose: If True, then intermediate console message will be displayed to indicate progress.

    Returns: the found sentences

    """
    log(text=f'🚶️  Walking over edges of the projected graph...', verbose=verbose)

    walker.walk(edges=edges)
    walk_file = walker.outfile

    log(text=f'📑️  Extracting Sentences...', verbose=verbose)
    return LineSentence(walk_file)
