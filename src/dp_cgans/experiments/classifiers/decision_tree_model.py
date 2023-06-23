from sklearn.tree import DecisionTreeClassifier

from dp_cgans.experiments.classifiers import ExperimentModel


class DecisionTreeModel(ExperimentModel):
    def _get_model(self):
        self._model = DecisionTreeClassifier()
