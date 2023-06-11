import os.path
import pickle

import pandas as pd

from dp_cgans.utils import Config
from dp_cgans.utils.files import load_csv, get_or_create_directory
from dp_cgans.utils.logging import log, LogLevel
from sentence_transformers import SentenceTransformer, util


class ZeroShotLearning:
    def __init__(self, config: str or Config):
        self.base_model: SentenceTransformer or None = None
        self.fitted_model: SentenceTransformer or None = None
        self.eval_model: SentenceTransformer or None = None

        if isinstance(config, str):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise Exception("Configuration could not be read.")

        self._init_verbose()

    def start(self):
        self._get_dataset()

        if self.config.get_nested('model', 'train'):
            self._train()
        else:
            self._load()

    def _get_dataset(self):
        dataset_directory = self.config.get_nested('dataset', 'files', 'directory')
        dataset_seen = self.config.get_nested('dataset', 'files', 'seen')
        dataset_unseen = self.config.get_nested('dataset', 'files', 'unseen')
        dataset_seen_path = f"{dataset_directory}/{dataset_seen}"
        dataset_unseen_path = f"{dataset_directory}/{dataset_unseen}"

        if not os.path.isfile(dataset_seen_path):
            raise IOError(
                f"The Seen Dataset located at \"{dataset_seen_path}\" does not exist, and can therefore not be processed.")

        if not os.path.isfile(dataset_unseen_path):
            raise IOError(
                f"The Unseen Dataset located at \"{dataset_unseen_path}\" does not exist, and can therefore not be processed.")

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

        self.labels = self.seen["classes"].unique()

    def _load(self):
        self.model = SentenceTransformer(self.config.get_nested('model', 'name'))

        model_directory = self.config.get_nested('model', 'files', 'directory')
        dataset_encoded = self.config.get_nested('model', 'files', 'model')
        dataset_encoded_path = f"{model_directory}/{dataset_encoded}"

        with open(dataset_encoded_path, "rb") as file:
            self.fitted_model = pickle.load(file)

    def _train(self):
        if not self.config.get_nested('model', 'train'):
            raise Exception("Training is disabled.")

        self.model = SentenceTransformer(self.config.get_nested('model', 'name'))

        converted = self.tab_to_text(self.seen["features"])

        self.fitted_model = self.model.encode(converted)

        model_directory = self.config.get_nested('model', 'files', 'directory')

        get_or_create_directory(model_directory)

        dataset_encoded = self.config.get_nested('model', 'files', 'model')
        dataset_encoded_path = f"{model_directory}/{dataset_encoded}"

        with open(dataset_encoded_path, "wb") as file:
            pickle.dump(self.fitted_model, file)

    def predict(self, embedding):
        S = util.pytorch_cos_sim(embedding, self.fitted_model)
        predicted_labels = []
        predicted_scores = []

        for i in range(embedding.shape[0]):
            label_scores = S[i].tolist()
            scored = sorted(
                zip(self.labels, label_scores),
                key=lambda x: x[1],
                reverse=True
            )

            pred, score = scored[0]
            threshold = self.config.get_nested('model', 'threshold')
            unknown_class = self.config.get_nested('model', 'unknown_class')

            if score < threshold:
                pred = unknown_class

            predicted_scores.append(scored)
            predicted_labels.append(pred)

            return predicted_labels, predicted_scores

    def tab_to_text(self, df):
        text_arr = []
        for idx_row, row in df.iterrows():
            text = "the given record "

            for col in df.columns:
                given_input = row[col]
                if given_input == 0:
                    text += f"does not have {col} "
                elif given_input == 1:
                    text += f"does have {col}"

            text_arr.append(text_arr)

        return text_arr

    def _init_verbose(self):
        self.verbose = self.config.get('verbose')

        if self.verbose is None or not isinstance(self.verbose, bool):
            self.verbose = False
