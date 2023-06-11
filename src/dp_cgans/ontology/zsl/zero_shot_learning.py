import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer, util, losses
from sklearn.metrics.pairwise import cosine_similarity

from dp_cgans.utils import Config
from dp_cgans.utils.data_types import load_config
from dp_cgans.utils.files import create_path, load_csv
from dp_cgans.utils.logging import log


class ZeroShotLearning:
    def __init__(self, config: str or Config):
        self._config = load_config(config)

        self._base_model: SentenceTransformer or None = None
        self._fitted_model: SentenceTransformer or None = None

        self._base_directory = self._config.get_nested('zsl', 'files', 'directory')

        self._init_verbose()

    def fit_or_load(self):
        if self._config.get_nested('zsl', 'model', 'load'):
            self.load()
        elif self._config.get_nested('zsl', 'model', 'save'):
            self.fit_and_save()
        else:
            self.fit()

    def fit_and_save(self):
        self.fit()
        self.save()

    def fit(self):
        if self._config.get_nested('zsl', 'model', 'load'):
            raise Exception("Can not train Zero-Shot Learning Classifier, when in loading mode.")

        self._get_dataset()

        log(text=f"Initiating Transformer...", verbose=self._verbose)
        self._base_model = SentenceTransformer(self._config.get_nested('zsl', 'model', 'name'))

        data = self._seen_features.values.tolist()

        if self._config.get_nested('zsl', 'model', 'prompt'):
            data = self._tab_to_text(self._seen_features)

        self._fitted_model = self._base_model.encode(data)

    def save(self):
        log(text=f"Saving model...", verbose=self._verbose)
        directory = self._config.get_nested('zsl', 'model', 'files', 'directory')

        # Prepend "_embedding_directory" the "directory", otherwise use path defined as "directory"
        if self._config.get_nested('zsl', 'model', 'files', 'use_root'):
            directory = create_path(self._base_directory, directory, create=True)

        log(text=f'Checking \"{directory}\" for datasets...', verbose=self._verbose)

        model_file = create_path(directory, self._config.get_nested('zsl', 'model', 'files', 'model'),
                                 create=True)

        log(text=f'Writing to file \"{model_file}\"...', verbose=self._verbose)

        with open(model_file, "wb") as file:
            pickle.dump(self._fitted_model, file)

        log(text=f'Model has been saved to \"{model_file}\".', verbose=self._verbose)

    def predict(self, data_to_predict):
        data = data_to_predict.values.tolist()
        if self._config.get_nested('zsl', 'model', 'prompt'):
            data = self._tab_to_text(data_to_predict)

        encodings = self._base_model.encode(data)

        predictions = []
        scores = []

        for embedding in encodings:
            pred, score = self._predict(embedding)
            predictions.append(pred)
            scores.append(score)

        if self._config.get_nested('zsl', 'model', 'files', 'results'):
            self._save_results(data_to_predict, predictions)

        return predictions, scores

    def _predict(self, embedding):
        score = cosine_similarity([embedding], self._fitted_model)[0]
        highest_score_index = np.argmax(score)
        threshold = self._config.get_nested('zsl', 'model', 'threshold')
        unknown_class = self._config.get_nested('zsl', 'model', 'unknown_class')

        if score[highest_score_index] < threshold:
            return unknown_class, score[highest_score_index]

        predicted_label = self._seen_classes.tolist()[highest_score_index]
        return predicted_label, score[highest_score_index]

    def _predict2(self, embedding):
        S = util.pytorch_cos_sim(embedding, self._fitted_model)
        predicted_labels = []
        predicted_scores = []

        for i in range(embedding.shape[0]):
            label_scores = S[i].tolist()
            scored = sorted(
                zip(self._seen_classes.unique(), label_scores),
                key=lambda x: x[1],
                reverse=True
            )

            pred, score = scored[0]
            threshold = self._config.get_nested('zsl', 'model', 'threshold')
            unknown_class = self._config.get_nested('zsl', 'model', 'unknown_class')

            if score < threshold:
                pred = unknown_class

            predicted_scores.append(scored)
            predicted_labels.append(pred)

            return predicted_labels, predicted_scores

    def _get_dataset(self):
        log(text=f"Loading Dataset...", verbose=self._verbose)

        directory = self._config.get_nested('dp_cgans', 'files', 'directory')

        log(text=f'Checking \"{directory}\" for datasets...', verbose=self._verbose)

        training_file = create_path(directory, self._config.get_nested('dp_cgans', 'files', 'input'),
                                    create=False)

        if not os.path.isfile(training_file):
            raise KeyError(f'We could not found a dataset at \"{training_file}\"')

        log(text=f"Extracting Features and Classes...", verbose=self._verbose)
        self._seen_features, self._seen_classes = load_csv(path=training_file,
                                                           class_header=self._config.get_nested('zsl', 'class_header'),
                                                           verbose=self._verbose)

        log(text=f"Dataset has been read!", verbose=self._verbose)

    def load(self):
        self._get_dataset()

        log(text=f"Initiating Transformer...", verbose=self._verbose)
        self._base_model = SentenceTransformer(self._config.get_nested('zsl', 'model', 'name'))

        log(text=f"Retrieving model...", verbose=self._verbose)
        directory = self._config.get_nested('zsl', 'model', 'files', 'directory')

        # Prepend "_embedding_directory" the "directory", otherwise use path defined as "directory"
        if self._config.get_nested('zsl', 'model', 'files', 'use_root'):
            directory = create_path(self._base_directory, directory, create=False)

        log(text=f'Checking \"{directory}\" for datasets...', verbose=self._verbose)

        model_file = create_path(directory, self._config.get_nested('zsl', 'model', 'files', 'model'),
                                 create=False)

        with open(model_file, "rb") as f:
            self._fitted_model = pickle.load(f)

    def _save_results(self, data, predictions):
        log(text=f"Saving model...", verbose=self._verbose)
        directory = self._config.get_nested('zsl', 'model', 'files', 'directory')

        # Prepend "_embedding_directory" the "directory", otherwise use path defined as "directory"
        if self._config.get_nested('zsl', 'model', 'files', 'use_root'):
            directory = create_path(self._base_directory, directory, create=True)

        log(text=f'Checking \"{directory}\" for datasets...', verbose=self._verbose)

        out_file = create_path(directory, self._config.get_nested('zsl', 'model', 'files', 'output'),
                               create=True)

        log(text=f'Writing to file \"{out_file}\"...', verbose=self._verbose)

        data['predicted'] = predictions
        data.to_csv(out_file, encoding="utf-8",
                    index=False,
                    header=True)
        log(text=f'Results has been saved to \"{out_file}\".', verbose=self._verbose)

    def _init_verbose(self):
        self._verbose = self._config.get_nested('verbose')

        if self._verbose is None or not isinstance(self._verbose, bool):
            self._verbose = False

    def _tab_to_text(self, df):
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
