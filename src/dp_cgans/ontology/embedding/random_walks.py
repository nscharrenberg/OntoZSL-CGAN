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
    """
    Enumeration class representing types of supported walkers.
    """
    DEEPWALK = "deepwalk"
    NODE2VEC = "node2vec"


def create_random_walker(walker_type: WalkerType = WalkerType.DEEPWALK, num_walks: int = 10, walk_length: int = 10,
                         outfile: any = "tmp/walks.txt", workers: int = 1, alpha: float = 0., p: float = 1.,
                         q: float = 1.,
                         verbose: bool = False) -> WalkingModel:
    """
    Creates a walker of the specified type to be used for walking on a graph.

    Args:
        walker_type: A WalkerType enum value representing the type of walker to create. Default: WalkerType.DEEPWALK.
        num_walks: An integer specifying the number of walks to perform. Default: 10.
        walk_length: An integer specifying the length of each walk. Default: 10.
        outfile: A string specifying the output file path for storing the walks. Default: "tmp/walks.txt".
        workers: An integer specifying the number of worker processes to use for parallel execution. Default: 1.
        alpha: A float specifying the probability of returning to the starting node in Node2Vec. Default: 0.0.
        p: A float specifying the return parameter in Node2Vec. Default: 1.0.
        q: A float specifying the in-out parameter in Node2Vec. Default: 1.0.
        verbose: A boolean indicating whether to enable verbose logging. Default: False.

    Returns:
        A WalkingModel object representing the created walker.
    """
    log(text=f'🧍️  Creating a Walker ({walker_type.value}) to be used for the graph to be walked on...',
        verbose=verbose)

    return walker_factory(method_name=walker_type.value, num_walks=num_walks, walk_length=walk_length, outfile=outfile,
                          workers=workers, alpha=alpha, p=p, q=q)


def pykeen_transe(dataset: PathDataset, dimension: int = 50, epochs: int = 10, batch_size: int = 32, seed: int = 42):
    """
    Creates and trains a TransE model using a given dataset.

    Args:
        dataset: A PathDataset object representing the ontology dataset.
        dimension: An integer specifying the embedding dimension. Default: 50.
        epochs: An integer specifying the number of training epochs. Default: 10.
        batch_size: An integer specifying the batch size for training. Default: 32.
        seed: An integer specifying the random seed. Default: 42.

    Returns:
        A trained KGEModel object representing the TransE model.

    Note: Current state does not seem to work properly.
    """
    triples_factory = Edge.as_pykeen(dataset.ontology, create_inverse_triples=True)
    pk_model = TransE(triples_factory=triples_factory, embedding_dim=dimension, random_seed=seed)
    model = KGEModel(triples_factory, pk_model, epochs=epochs, batch_size=batch_size)
    model.train()

    return model


def walk(edges: List[Edge], walker: WalkingModel, verbose: bool = False) -> LineSentence:
    """
    Performs a random walk over the edges of a projected graph using a walker.

    Args:
        edges: A list of Edge objects representing the edges of the projected graph.
        walker: A WalkingModel object representing the walker to use for the walk.
        verbose: A boolean indicating whether to enable verbose logging. Default: False.

    Returns:
        A LineSentence object representing the resulting walk as a sequence of sentences.
    """
    log(text=f'🚶️  Walking over edges of the projected graph...', verbose=verbose)

    walker.walk(edges=edges)
    walk_file = walker.outfile

    log(text=f'📑️  Extracting Sentences...', verbose=verbose)
    return LineSentence(walk_file)
