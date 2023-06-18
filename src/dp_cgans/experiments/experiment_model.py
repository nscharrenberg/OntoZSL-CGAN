import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from dp_cgans.utils import Config
from dp_cgans.utils.files import get_or_create_directory, create_path


class ExperimentModel:
    def __init__(self, config: str or Config):
        self.score = None
        self._model: BaseEstimator or None = None
        if isinstance(config, str):
            self._config = Config(config)
        elif isinstance(config, Config):
            self._config = config
        else:
            raise Exception("Configuration could not be read.")

        self._load_dataset()
        self._get_model()

    def _load_dataset(self):
        header_class = self._config.get_nested('header_class')

        directory = None

        if self._config.get_nested('dataset', 'use_root'):
            directory = get_or_create_directory(self._config.get_nested('dataset', 'directory'), error=True)

        training_path = create_path(directory, self._config.get_nested('dataset', 'training'), create=False)

        training_data = pd.read_csv(training_path)

        training = {
            "features": training_data.drop(header_class, axis=1),
            "labels": training_data[header_class]
        }

        self._label_encoder = LabelEncoder()

        testing_path = self._config.get_nested('dataset', 'testing')

        if testing_path is not None:
            testing_path = create_path(directory, testing_path)

            testing_data = pd.read_csv(testing_path)

            testing = {
                "features": testing_data.drop(header_class, axis=1),
                "labels": testing_data[header_class]
            }

            self._data = {
                "training": training,
                "testing": testing
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(training['features'], training['labels'],
                                                                test_size=self._config.get_nested('test_size'),
                                                                random_state=self._config.get_nested('seed'))

            self._data = {
                "training": {
                    "features": X_train,
                    "labels": y_train
                },
                "testing": {
                    "features": X_test,
                    "labels": y_test
                }
            }

        self._data['training']['labels'] = self._label_encoder.fit_transform(self._data['training']['labels'])
        self._data['testing']['labels'] = self._label_encoder.fit_transform(self._data['testing']['labels'])

    def _get_model(self):
        raise NotImplementedError("An unfinished model is being run.")

    def fit(self):
        self._model.fit(self._data['training']['features'], self._data['training']['labels'])

    def _predict(self):
        return self._model.predict(self._data['testing']['features'])

    def evaluate(self):
        y_pred = self._predict()
        y_true = self._data['testing']['labels']

        df = pd.DataFrame(self._data['testing']['features'])
        df['y_pred'] = y_pred
        df['y_true'] = y_true

        # Accuracy of the Model
        accuracy = accuracy_score(y_true, y_pred)

        # F1 Score of the Model
        f1 = f1_score(y_true, y_pred, average='weighted')

        n_classes = len(self._label_encoder.classes_)
        y_test_bin = label_binarize(y_true, classes=range(n_classes))
        y_pred_bin = label_binarize(y_pred, classes=range(n_classes))

        # Weighted AUC
        weighted_auc = roc_auc_score(y_test_bin, y_pred_bin, average='weighted')

        # Macro AUC for each class independently
        auc_scores = []
        for i in range(n_classes):
            auc = roc_auc_score(y_test_bin[:, i], y_pred_bin[:, i])
            auc_scores.append(auc)

        macro_auc = sum(auc_scores) / n_classes

        # Micro AUC aggregating TP/FP/FN/TN
        micro_auc = roc_auc_score(y_test_bin, y_pred_bin, average='micro')

        result_directory = self._config.get_nested('results', 'directory')
        result_file = create_path(result_directory, self._config.get_nested('results', 'file'), create=True)
        data_file = create_path(result_directory, 'data.csv', create=False)

        self.score = {
            "accuracy": accuracy,
            "f1": f1,
            "weighted_auc": weighted_auc,
            "macro_auc": macro_auc,
            "micro_auc": micro_auc
        }

        result_df = pd.DataFrame([self.score], columns=['accuracy', 'f1', 'weighted_auc', 'macro_auc', 'micro_auc'])
        result_df.to_csv(result_file, encoding="utf-8",
                         index=False,
                         header=True)

        df.to_csv(data_file, encoding="utf-8",
                         index=False,
                         header=True)
