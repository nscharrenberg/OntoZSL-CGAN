"""This embedding method is used based on the MOWL Documentation"""

import mowl

# Needs to be instantiated before other MOWL imports are triggered.
mowl.init_jvm("20g")

from mowl.datasets import PathDataset
from src.dp_cgans.embeddings import create_projection, ProjectionType, project, create_random_walker, WalkerType, walk
from src.dp_cgans.embeddings.utils import log
from gensim.models import Word2Vec


def embed(data: str, model_path: str, dim: int, verbose=True,
          projection_type: ProjectionType = ProjectionType.OWL2VECSTAR,
          bidirectional_taxonomy=False, include_literals=False,
          only_taxonomy=False, use_taxonomy=True, relations=None,
          walker_type: WalkerType = WalkerType.DEEPWALK, num_walks: int = 10,
          walk_length: int = 10, outfile: any = None, workers: int = 4, alpha: float = 0.1,
          p: float = 1., q: float = 1., epochs: int = 10, window: int = 5, min_count: int = 1):
    """
    Build an Ontology Embedding Model by:
    1.  Projecting the given Semantic Ontology Language dataset
    2.  Walking over the projected edges to create sentences
    3.  Convert it into a vectorized model
    Args:
        data: Path to a Semantic Ontology Language Dataset (.owl)
        model_path: Path to where the model should be stored (.model)
        dim: Dimensionality of the Word Vectors used to embed the model
        verbose: If True, then intermediate console message will be displayed to indicate progress.
        projection_type: The Projection Method that is used to project the semantic information into a graph \
            the possible methods are owl2vecstar, dl2vec, taxonomy and taxonomy_rels
        bidirectional_taxonomy: If true, then per each subClass edge one superClass edge will be generated \
            Can only be True when use_taxonomy is True.
        include_literals: If true, then the graph will also include triples involving data property assertions \
            and annotations, for owl2vecstar
        only_taxonomy: If true, then the projection will only include subClass edges, for owl2vecstar
        use_taxonomy: If true, then taxonomies will be used. Otherwise, the relations parameter should be true, \
            for taxonomy_rels
        relations: Contains a list of relations in string format, for taxonomy_rels \
            Can only be non-empty when use_taxonomy is False.
        walker_type: The walker Method that is used to learn the representations for vertices in the projected graph \
            the possible methods are deepwalk and node2vec
        num_walks: The number of walks
        walk_length: The length of a walk
        outfile: The output File of the Walk
        workers: The amount of workers
        alpha: The probability of a restart, for DeepWalk
        p: The Return Hyperparameter, for Node2Vec
        q: The In-Out Hyperparameter, for Node2Vec
        epochs: Number of iterations (epochs) over the sentences, for Word2Vec
        window: Maximum distance between the current and predicted word within a sentence, for Word2Vec
        min_count: Ignores all words with total frequency lower than this, for Word2Vec

    Returns: A trained embedded ontology model.
    """
    # Load the dataset
    log(text=f'📖️  Loading Dataset from {data}...', verbose=verbose)
    dataset = PathDataset(data)

    # Instantiate projector type to be used to project the data into a graph
    projector = create_projection(projection_type=projection_type, bidirectional_taxonomy=bidirectional_taxonomy,
                                  include_literals=include_literals, only_taxonomy=only_taxonomy,
                                  use_taxonomy=use_taxonomy, relations=relations, verbose=verbose)

    edges = project(dataset=dataset, projector=projector, verbose=verbose)

    walker = create_random_walker(walker_type=walker_type, num_walks=num_walks, walk_length=walk_length,
                                  outfile=outfile, workers=workers, alpha=alpha, p=p, q=q, verbose=verbose)

    sentences = walk(edges=edges, walker=walker, verbose=verbose)

    log(text=f'🏗️  Training the model...', verbose=verbose)
    model = Word2Vec(sentences=sentences, vector_size=dim, epochs=epochs, window=window, min_count=min_count)

    log(text=f'💾️  Saving the model to {model_path}...', verbose=verbose)
    model.save(model_path)

    log(text=f'✅️The ontology dataset has been successfully embedded!', verbose=verbose)

    return model


def load_embedding(model_path):
    """
    Loads in a pretrained embedded ontology model
    Args:
        model_path: Path to an Embedded Ontology Model file (.model)

    Returns: The loaded model
    """
    return Word2Vec.load(model_path)
