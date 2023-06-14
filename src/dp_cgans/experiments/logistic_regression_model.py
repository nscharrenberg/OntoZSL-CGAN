from sklearn.linear_model import LogisticRegression

from dp_cgans.experiments.experiment_model import ExperimentModel


class LogisticRegressionModel(ExperimentModel):
    def _get_model(self):
        self._model = LogisticRegression()
