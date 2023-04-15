"""This embedding method is based on the MOWL OWL2Vec* and DeepWalk Documentation"""

import mowl

# Needs to be instantiated before other MOWL imports are triggered.
mowl.init_jvm("20g")

from mowl.datasets import PathDataset
from mowl.projection import OWL2VecStarProjector
from mowl.walking import DeepWalk

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import typer

from mowl.visualization import TSNE

cli = typer.Typer()


def main(data, model_path, dim, load_model, graph):
    if load_model:
        model = load(model_path)
    else:
        model = embed(data, model_path, dim)

    if graph:
        visualize(model)

    return model


def embed(data, model_path, dim):
    """Using Owl2Vec* and DeepWalk to generate an ontological model"""

    # Load the dataset
    print("Loading Dataset...")
    dataset = PathDataset(data)

    # Create an Owl2Vec* graph by utilizing the Projector
    # Source: https://mowl.readthedocs.io/en/latest/graphs/projection.html#owl2vec
    print("Creating Owl2Vec* Projection...")
    projector = OWL2VecStarProjector(
        bidirectional_taxonomy=True,
        include_literals=False,
        only_taxonomy=False
    )

    # Generate random walks using DeepWalk
    # Source: https://mowl.readthedocs.io/en/latest/graphs/random_walks.html#deepwalk
    print("Extracting Edges...")
    edges = projector.project(dataset.ontology)

    print("Walking over Edges...")
    walker = DeepWalk(10, 10, 0.1, workers=4)
    walker.walk(edges)
    walk_file = walker.outfile

    print("Extracting Sentences...")
    sentences = LineSentence(walk_file)

    # Create a Word2Vec model after performing the random walks and store this as a model
    print("Building Model...")
    model = Word2Vec(sentences, vector_size=dim, epochs=10, window=5, min_count=1, workers=4)
    model.save(model_path)

    print(f"Model Saved in {model_path}!")

    return model


def load(model):
    """Load an existing ontological model"""
    return Word2Vec.load(model)


def visualize(model):
    """Visualize the embedded model using t-SNE"""
    embeddings = []
    keys = []
    labels = []
    current_count = 0

    print("Extracting Embeddings and Labels. This may take some time...")
    for key in model.wv.index_to_key:
        embeddings.append(model.wv[key])
        keys.append(key)
        labels.append(current_count)
        current_count = current_count + 1

        if current_count == 100:
            break

    print("Instantiating t-SNE...")
    tsne = TSNE(dict(zip(labels, embeddings)), dict(zip(labels, keys)))
    tsne.generate_points(250, workers=4)

    print("Visualizing...")
    tsne.show()
    print("Visualization Completed!")
