import os.path
from typing import List

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from mowl.datasets import PathDataset
from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
from mowl.projection import Edge
from mowl.evaluation.base import CosineSimilarity

from dp_cgans.ontology.embedding.projections import create_projection, ProjectionType
from dp_cgans.ontology.embedding.random_walks import pykeen_transe, create_random_walker, WalkerType
from dp_cgans.utils import Config

from dp_cgans.utils.files import recode
from dp_cgans.utils.logging import log


class Embedding:
    def __init__(self, config: str or Config) -> None:
        if isinstance(config, str):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise Exception("Configuration could not be read.")

        self.dataset = None
        self.model = None

        self._init_verbose()

    def _init_verbose(self):
        self.verbose = self.config.get('verbose')

        if self.verbose is None or not isinstance(self.verbose, bool):
            self.verbose = False

    def start(self) -> Word2Vec:
        self._load_dataset()

        if self.dataset is None:
            raise Exception("Dataset has not been loaded.")

        self._project()
        self._walk()
        self._save()

        if self.config.get('evaluate'):
            self.evaluate()

        log(text=f'Embedding has been completed!', verbose=self.verbose)

        return self.model

    def load_config(self, config: str) -> Config:
        self.config = Config(config)
        self.verbose = self.config.get('verbose')

        return self.config

    def load(self, model: str = None) -> Word2Vec:
        if model is None:
            model = f"{self.config.get_nested('model', 'directory')}/{self.config.get_nested('model', 'file')}"

        if not os.path.isfile(model):
            raise Exception("Model does not exist. Make sure the path is correct.")

        self._load_dataset()

        self.model = Word2Vec.load(model)
        return self.model

    def _load_dataset(self) -> PathDataset:
        log(text=f'Loading Dataset...', verbose=self.verbose)
        directory = self.config.get('dataset_directory')

        log(text=f'Checking \"{directory}\" for datasets...', verbose=self.verbose)

        training_path = f"{directory}/{self.config.get_nested('datasets', 'training')}"

        if not os.path.isfile(training_path):
            raise KeyError(f'We could not found a dataset at \"{training_path}\"')

        log(text=f'Found a Dataset for training at \"{training_path}\"', verbose=self.verbose)

        testing_path = f"{directory}/{self.config.get_nested('datasets', 'testing')}"

        if not os.path.isfile(testing_path):
            testing_path = None
        else:
            log(text=f'Found a Dataset for testing at \"{testing_path}\"', verbose=self.verbose)

        validation_path = f"{directory}/{self.config.get_nested('datasets', 'validation')}"

        if not os.path.isfile(validation_path):
            validation_path = None
        else:
            log(text=f'Found a Dataset for validation at \"{validation_path}\"', verbose=self.verbose)

        self.dataset = PathDataset(ontology_path=training_path, validation_path=validation_path, testing_path=testing_path)

        log(text=f'Dataset has been created!', verbose=self.verbose)

        return self.dataset

    def _project(self) -> List[Edge]:
        log(text=f'Initializing Graph Projection...', verbose=self.verbose)
        self.projector = create_projection(
            projection_type=ProjectionType(self.config.get_nested('projection', 'type')),
            bidirectional_taxonomy=self.config.get_nested('projection', 'bidirectional_taxonomy'),
            include_literals=self.config.get_nested('projection', 'include_literals'),
            only_taxonomy=self.config.get_nested('projection', 'only_taxonomy'),
            use_taxonomy=self.config.get_nested('projection', 'use_taxonomy'),
            relations=self.config.get_nested('projection', 'relations'),
            verbose=self.verbose
        )

        log(text=f'Projecting Graph...', verbose=self.verbose)

        self.edges = self.projector.project(self.dataset.ontology)

        log(text=f'Graph has been projected!', verbose=self.verbose)
        return self.edges

    def _walk(self) -> LineSentence:
        log(text=f'Initializing Edge Walking...', verbose=self.verbose)

        outfile_directory = self.config.get_nested('embedding', 'outfile', 'directory')
        outfile_name = self.config.get_nested('embedding', 'outfile', 'file')
        outfile_path = f"{outfile_directory}/{outfile_name}"

        if not os.path.exists(outfile_directory):
            log(text=f'Creating outfile directory...', verbose=self.verbose)
            os.makedirs(outfile_directory)
            log(text=f'Outfile directory has been created at \"{outfile_directory}\"', verbose=self.verbose)

        self.walker = create_random_walker(
            walker_type=WalkerType(self.config.get_nested('embedding', 'type')),
            num_walks=self.config.get_nested('embedding', 'num_walks'),
            walk_length=self.config.get_nested('embedding', 'walk_length'),
            workers=self.config.get_nested('embedding', 'workers'),
            alpha=self.config.get_nested('embedding', 'alpha'),
            p=self.config.get_nested('embedding', 'p'),
            q=self.config.get_nested('embedding', 'q'),
            outfile=outfile_path,
            verbose=self.verbose
        )

        log(text=f'Walking Edges...', verbose=self.verbose)
        self.sentences = self.walker.walk(edges=self.edges)
        walks = self.walker.outfile

        # mOWL seems to save the outfile in ANSI format, while LineSentence expects utf-8
        target_file = f"{outfile_directory}/recoded_{outfile_name}"
        recode(source=walks, target=target_file, decoding="ANSI", encoding="utf-8", verbose=self.verbose)

        log(text=f'Creating Sentences...', verbose=self.verbose)
        self.sentences = LineSentence(target_file)

        log(text=f'Sentences have been created!', verbose=self.verbose)

        return self.sentences

    def _save(self) -> Word2Vec:
        log(text=f'Preparing model to be saved...', verbose=self.verbose)
        model_directory = self.config.get_nested('model', 'directory')
        model_path = f"{model_directory}/{self.config.get_nested('model', 'file')}"

        if not os.path.exists(model_directory):
            log(text=f'Creating model directory...', verbose=self.verbose)
            os.makedirs(model_directory)
            log(text=f'Model directory has been created at \"{model_directory}\"', verbose=self.verbose)

        self.model = Word2Vec(
            sentences=self.sentences,
            vector_size=self.config.get_nested('word2vec', 'dimensions'),
            epochs=self.config.get_nested('word2vec', 'epochs'),
            window=self.config.get_nested('word2vec', 'window'),
            min_count=self.config.get_nested('word2vec', 'min_count')
        )

        log(text=f'Saving model...', verbose=self.verbose)
        self.model.save(model_path)

        text_file_name = self.config.get_nested('word2vec', 'text_file')
        if text_file_name is not None:
            log(text=f'Saving model as text...', verbose=self.verbose)
            text_path = f"{model_directory}/{text_file_name}"
            self.model.wv.save_word2vec_format(text_path, binary=False)

        log(text=f'Model has been saved!', verbose=self.verbose)

        return self.model

    def evaluate(self):
        if not self.dataset:
            raise Exception("Dataset is not loaded")

        if not self.model:
            raise Exception("Model is not loaded")

        classes = self.dataset.evaluation_classes
        evaluation_edges = self.projector.project(self.dataset.testing)
        filtering_edges = self.projector.project(self.dataset.ontology)

        vectors = self.model.wv
        evaluator = EmbeddingsRankBasedEvaluator(vectors,
                                                 evaluation_edges,
                                                 CosineSimilarity,
                                                 training_set=filtering_edges,
                                                 head_entities=classes.as_str,
                                                 tail_entities=classes.as_str,
                                                 device='cpu'
                                                 )
        evaluator.evaluate(show=True)




