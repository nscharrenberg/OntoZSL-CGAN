from enum import Enum
from typing import List

from gensim.models.word2vec import LineSentence
from mowl.projection import Edge

from src.dp_cgans.embeddings.utils import log
from mowl.projection.base import ProjectionModel
from mowl.walking import walker_factory
from mowl.walking.walking import WalkingModel
from mowl.datasets import PathDataset


class WalkerType(Enum):
    DEEPWALK = "deepwalk"
    NODE2VEC = "node2vec"


def create_random_walker(walker_type: WalkerType, num_walks: int, walk_length: int, outfile: any = None,
                         workers: int = 1, alpha: float = 0., p: float = 1., q: float = 1., verbose: bool = False) -> WalkingModel:
    log(text=f'🧍️  Creating a Walker ({walker_type.value}) to be used for the graph to be walked on...',
        verbose=verbose)
    return walker_factory(method_name=walker_type.value, num_walks=num_walks, walk_length=walk_length, outfile=outfile,
                          workers=workers, alpha=alpha, p=p, q=q)


def walk(edges: List[Edge], walker: WalkingModel, verbose: bool = False):
    log(text=f'🚶️  Walking over edges of the projected graph...',
        verbose=verbose)
    walker.walk(edges=edges)
    walk_file = walker.outfile

    log(text=f'📑️  Extracting Sentences...',
        verbose=verbose)
    return LineSentence(walk_file)





