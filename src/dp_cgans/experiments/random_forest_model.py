from sklearn.ensemble import RandomForestClassifier

from dp_cgans.experiments.experiment_model import ExperimentModel


class RandomForestModel(ExperimentModel):
    def _get_model(self):
        self._model = RandomForestClassifier()
