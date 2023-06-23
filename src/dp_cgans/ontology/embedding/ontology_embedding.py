import os
from typing import List

import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from mowl.datasets import PathDataset
from mowl.projection import Edge

from dp_cgans.ontology.embedding.projections import create_projection, ProjectionType
from dp_cgans.ontology.embedding.random_walks import create_random_walker, WalkerType
from dp_cgans.utils import Config
from dp_cgans.utils.data_types import load_config
from dp_cgans.utils.files import create_path, get_or_create_directory, recode
from dp_cgans.utils.logging import log


class OntologyEmbedding:
    """
    A class for performing ontology embedding using Word2Vec.

    The class provides methods for fitting the embedding model, saving and loading the model,
    obtaining the embedding vector for an entity, and retrieving the corresponding IRI.

    """

    def __init__(self, config: str or Config):
        """
        Initialize the OntologyEmbedding object.

        Args:
            config (str or Config): Path to the configuration file or an instance of Config.

        """
        self._config = load_config(config)

        self._embedding_directory = get_or_create_directory(self._config.get_nested('embedding', 'files', 'directory'),
                                                            error=True)

        self.embed_size = self._config.get_nested('embedding', 'word2vec', 'dimensions')
        self.model = None
        self._dict = None
        self._init_verbose()

    def fit_or_load(self) -> Word2Vec:
        """
        Fit the embedding model or load a pre-trained model.

        Returns:
            Word2Vec: The trained or loaded Word2Vec model.

        """
        if self._config.get_nested('embedding', 'model', 'load'):
            self.load()
        elif self._config.get_nested('embedding', 'model', 'save'):
            self.fit_and_save()
        else:
            self.fit()

        log(text=f'Ontology Embedding Model is Initialized (trained, saved and/or loaded)', verbose=self._verbose)

        return self.model

    def fit_and_save(self) -> Word2Vec:
        """
        Fit the embedding model and save it.

        Returns:
            Word2Vec: The trained Word2Vec model.

        """
        self.fit()
        self.save()

        return self.model

    def fit(self) -> Word2Vec:
        """
        Fit the embedding model.

        Returns:
            Word2Vec: The trained Word2Vec model.

        Raises:
            Exception: If the dataset has not been loaded.

        """
        self._load_dataset()

        if self._dataset is None:
            raise Exception("Dataset has not been loaded.")

        self._project()
        self._random_walks()
        self._build_model()
        self._generate_dict()

        log(text=f'Embedding has been completed!', verbose=self._verbose)

        return self.model

    def save(self):
        """
        Save the trained Word2Vec model.

        Raises:
            Exception: If the model has not been trained before saving.

        """
        log(text=f'Preparing model to be saved...', verbose=self._verbose)

        if self.model is None:
            raise Exception("Model must have been trained, before it can be exported.")

        model_directory = self._config.get_nested('embedding', 'model', 'directory')

        if self._config.get_nested('embedding', 'model', 'use_root'):
            model_directory = create_path(self._embedding_directory, model_directory, create=True)

        model_file = create_path(model_directory, self._config.get_nested('embedding', 'model', 'file'),
                                 create=True)

        log(text=f'Model directory has been created at \"{model_directory}\"', verbose=self._verbose)

        log(text=f'Saving model...', verbose=self._verbose)
        self.model.save(model_file)

        if self._config.get_nested('embedding', 'model', 'save_key_vectors'):
            txt_file = create_path(model_directory, f"{self._config.get_nested('embedding', 'model', 'file')}.kv", create=False)
            self.model.wv.save(txt_file)

        log(text=f'Model has been saved to \"{model_file}\"!', verbose=self._verbose)

    def load(self) -> Word2Vec:
        """
        Load a pre-trained Word2Vec model.

        Returns:
            Word2Vec: The loaded Word2Vec model.

        """
        model_directory = self._config.get_nested('embedding', 'model', 'directory')

        if self._config.get_nested('embedding', 'model', 'use_root'):
            model_directory = create_path(self._embedding_directory, model_directory, create=False)

        model_file = create_path(model_directory, self._config.get_nested('embedding', 'model', 'file'),
                                 create=False)

        self._load_dataset()
        self.model = Word2Vec.load(model_file)
        self._generate_dict()

        log(text=f'Ontology Embedding has been read.', verbose=self._verbose)

        return self.model

    def _project(self) -> List[Edge]:
        """
        Perform graph projection.

        Returns:
            List[Edge]: The projected edges.

        """
        log(text=f'Initializing Graph Projection...', verbose=self._verbose)
        self._projector = create_projection(
            projection_type=ProjectionType(self._config.get_nested('embedding', 'projection', 'type')),
            bidirectional_taxonomy=self._config.get_nested('embedding', 'projection', 'bidirectional_taxonomy'),
            include_literals=self._config.get_nested('embedding', 'projection', 'include_literals'),
            only_taxonomy=self._config.get_nested('embedding', 'projection', 'only_taxonomy'),
            use_taxonomy=self._config.get_nested('embedding', 'projection', 'use_taxonomy'),
            relations=self._config.get_nested('embedding', 'projection', 'relations'),
            verbose=self._verbose
        )

        log(text=f'Projecting Graph...', verbose=self._verbose)

        self._edges = self._projector.project(self._dataset.ontology)

        log(text=f'Graph has been projected!', verbose=self._verbose)

        return self._edges

    def _random_walks(self):
        """
         Perform random walks on the graph.

         """
        log(text=f'Initializing Random Walks...', verbose=self._verbose)

        out_directory = self._config.get_nested('embedding', 'random_walks', 'outfile', 'directory')

        # Prepend "_embedding_directory" the "directory", otherwise use path defined as "directory"
        if self._config.get_nested('embedding', 'random_walks', 'outfile', 'use_root'):
            out_directory = create_path(self._embedding_directory, out_directory, create=True)

        out_file = create_path(out_directory, self._config.get_nested('embedding', 'random_walks', 'outfile', 'file'),
                               create=True)

        backup_out_file = create_path(out_directory,
                                      f"{self._config.get_nested('embedding', 'random_walks', 'outfile', 'file')}.bak",
                                      create=True)

        log(text=f'Outfile directory has been created at \"{out_directory}\"', verbose=self._verbose)

        self._walker = create_random_walker(
            walker_type=WalkerType(self._config.get_nested('embedding', 'random_walks', 'type')),
            num_walks=self._config.get_nested('embedding', 'random_walks', 'num_walks'),
            walk_length=self._config.get_nested('embedding', 'random_walks', 'walk_length'),
            workers=self._config.get_nested('embedding', 'random_walks', 'workers'),
            alpha=self._config.get_nested('embedding', 'random_walks', 'alpha'),
            p=self._config.get_nested('embedding', 'random_walks', 'p'),
            q=self._config.get_nested('embedding', 'random_walks', 'q'),
            outfile=backup_out_file,
            verbose=self._verbose
        )

        log(text=f'Walking Edges...', verbose=self._verbose)
        self._walker.walk(edges=self._edges)
        walks = self._walker.outfile

        # mOWL seems to save the outfile in ANSI format, while LineSentence expects utf-8
        recode(source=walks, target=out_file, decoding="ANSI", encoding="utf-8", verbose=self._verbose)

        os.remove(backup_out_file)

        log(text=f'Creating Sentences...', verbose=self._verbose)
        self._sentences = LineSentence(out_file)

        log(text=f'Sentences have been created!', verbose=self._verbose)

        return self._sentences

    def _build_model(self) -> Word2Vec:
        """
        Build the Word2Vec model.

        Returns:
            Word2Vec: The trained Word2Vec model.

        """
        log(text=f'Building Word2Vec Model...', verbose=self._verbose)

        self.model = Word2Vec(
            sentences=self._sentences,
            vector_size=self._config.get_nested('embedding', 'word2vec', 'dimensions'),
            epochs=self._config.get_nested('embedding', 'word2vec', 'epochs'),
            window=self._config.get_nested('embedding', 'word2vec', 'window'),
            min_count=self._config.get_nested('embedding', 'word2vec', 'min_count')
        )

        return self.model

    def _load_dataset(self) -> PathDataset:
        """
        Load the ontology dataset.

        Returns:
            PathDataset: The loaded PathDataset object.

        Raises:
            KeyError: If the dataset file is not found.

        """
        log(text=f"Loading Dataset...", verbose=self._verbose)
        directory = self._config.get_nested('embedding', 'datasets', 'directory')

        # Prepend "_embedding_directory" the "directory", otherwise use path defined as "directory"
        if self._config.get_nested('embedding', 'datasets', 'use_root'):
            directory = create_path(self._embedding_directory, directory, create=False)

        log(text=f'Checking \"{directory}\" for datasets...', verbose=self._verbose)

        training_file = create_path(directory, self._config.get_nested('embedding', 'datasets', 'training'),
                                    create=False)

        if not os.path.isfile(training_file):
            raise KeyError(f'We could not found a dataset at \"{training_file}\"')

        log(text=f'Found a Dataset for training at \"{training_file}\"', verbose=self._verbose)

        testing_file = create_path(directory, self._config.get_nested('embedding', 'datasets', 'testing'),
                                   create=False)

        if testing_file is None:
            log(text=f'No testing ontology specified.', verbose=self._verbose)
        else:
            log(text=f'Found a Dataset for testing at \"{testing_file}\"', verbose=self._verbose)

        validation_file = create_path(directory, self._config.get_nested('embedding', 'datasets', 'validation'),
                                      create=False)

        if validation_file is None:
            log(text=f'No validation ontology specified.', verbose=self._verbose)
        else:
            log(text=f'Found a Dataset for validation at \"{validation_file}\"', verbose=self._verbose)

        self._dataset = PathDataset(ontology_path=training_file, testing_path=testing_file,
                                    validation_path=validation_file)
        log(text=f'Dataset has been loaded!', verbose=self._verbose)

        return self._dataset

    def _generate_dict(self):
        """
        Generate the entity-to-IRI dictionary.

        """
        log(text=f'Preparing to read IRI dictionaries...', verbose=self._verbose)
        references = self._config.get_nested('embedding', 'references')
        self._dict = {}

        for reference in references:
            log(text=f'Reading \"{reference}\"', verbose=self._verbose)
            if not os.path.exists(reference):
                raise Exception(f"Could not find IRI Reference \"{reference}\"")

            with open(reference) as f:
                for line in f:
                    (entity, iri) = line.strip().split(";")
                    self._dict[entity] = iri

        log(text=f'All IRI Dictionaries have been read.', verbose=self._verbose)

    def get_iri(self, entity):
        """
        Get the IRI corresponding to an entity.

        Args:
            entity (str): The entity for which to retrieve the IRI.

        Returns:
            str: The corresponding IRI.

        Raises:
            Exception: If the IRI dictionary is empty or the entity is not found.

        """
        if self._dict is None or not self._dict:
            raise Exception("IRI Dictionary is empty, while trying to find an IRI.")

        return self._dict.get(entity, None)

    def get_embedding(self, entity):
        """
        Get the embedding vector for an entity.

        Args:
            entity (str): The entity for which to retrieve the embedding.

        Returns:
            numpy.ndarray: The embedding vector.

        Raises:
            Exception: If the model is not trained, the IRI dictionary is empty, or the entity is not found.

        """
        if self.model is None:
            raise Exception("Can not get an embedding from an untrainde model.")

        if self._dict is None or not self._dict:
            raise Exception("IRI Dictionary is empty, while trying to find an IRI.")

        iri = self.get_iri(entity)

        if iri is not None:
            return self.model.wv[iri]

        return np.zeros(self.embed_size)

    def _init_verbose(self):
        self._verbose = self._config.get_nested('verbose')

        if self._verbose is None or not isinstance(self._verbose, bool):
            self._verbose = False
