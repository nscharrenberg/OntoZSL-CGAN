from sklearn.tree import DecisionTreeClassifier

from dp_cgans.experiments.experiment_model import ExperimentModel


class DecisionTreeModel(ExperimentModel):
    def _get_model(self):
        self._model = DecisionTreeClassifier()
