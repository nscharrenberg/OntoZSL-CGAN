import os.path

import numpy as np
from sklearn.metrics import confusion_matrix, silhouette_score

from dp_cgans.utils import Config
from dp_cgans.utils.files import load_csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dp_cgans.utils.logging import log


class ZeroShotLearning:
    def __init__(self, config: str or Config):
        self.embedding = None
        if isinstance(config, str):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise Exception("Configuration could not be read.")

        self._init_verbose()

    def start(self):
        self._get_dataset()

        model = SentenceTransformer('bert-base-uncased')
        seen_embedding = model.encode(self.seen["features"].values.tolist())
        unseen_embedding = model.encode(self.unseen["features"].values.tolist())

        self.embedding = {
            "seen": seen_embedding,
            "unseen": unseen_embedding
        }

        predictions = []

        for embedding in unseen_embedding:
            score = cosine_similarity([embedding], seen_embedding)[0]
            predicted_label = self.seen["classes"].tolist()[np.argmax(score)]
            predictions.append(predicted_label)

        if self.config.get_nested('evaluate', 'confusion_matrix'):
            log(f"Creating confusion matrix... (this doesn't make much sense for clustering, but helps us manually validate on very small datasets.)",
                verbose=self.verbose)
            matrix = confusion_matrix(self.unseen["classes"].tolist(), predictions)
            log(f"Confusion Matrix: \n {matrix}")

        if self.config.get_nested('evaluate', 'silhouette'):
            log(f"Computing Silhouette Average...",
                verbose=self.verbose)
            silhouette_avg = silhouette_score(unseen_embedding, predictions)
            log(f"Silhouette Average: \n {silhouette_avg}")

    def _get_dataset(self):
        dataset_directory = self.config.get_nested('dataset', 'files', 'directory')
        dataset_seen = self.config.get_nested('dataset', 'files', 'seen')
        dataset_unseen = self.config.get_nested('dataset', 'files', 'unseen')
        dataset_seen_path = f"{dataset_directory}/{dataset_seen}"
        dataset_unseen_path = f"{dataset_directory}/{dataset_unseen}"

        if not os.path.isfile(dataset_seen_path):
            raise IOError(f"The Seen Dataset located at \"{dataset_seen_path}\" does not exist, and can therefore not be processed.")

        if not os.path.isfile(dataset_unseen_path):
            raise IOError(f"The Unseen Dataset located at \"{dataset_unseen_path}\" does not exist, and can therefore not be processed.")

        seen_features, seen_classes = load_csv(path=dataset_seen_path,
                                               class_header=self.config.get_nested('dataset', 'class_header'),
                                               verbose=self.verbose)
        unseen_features, unseen_classes = load_csv(path=dataset_unseen_path,
                                                   class_header=self.config.get_nested('dataset', 'class_header'),
                                                   verbose=self.verbose)

        self.seen = {
            "features": seen_features,
            "classes": seen_classes
        }

        self.unseen = {
            "features": unseen_features,
            "classes": unseen_classes
        }

    def _init_verbose(self):
        self.verbose = self.config.get('verbose')

        if self.verbose is None or not isinstance(self.verbose, bool):
            self.verbose = False
