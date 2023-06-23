from sklearn.ensemble import RandomForestClassifier

from dp_cgans.experiments.classifiers import ExperimentModel


class RandomForestModel(ExperimentModel):
    def _get_model(self):
        self._model = RandomForestClassifier(random_state=42)
