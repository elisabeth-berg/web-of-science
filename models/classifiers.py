from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from models.loader import *


class ClassyModel(object):

    def __init__(self, model):
        self.model = model

    def split_n_fit(self, X, y, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=4618)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_hat = self.model.predict(self.X_test)

    def score(self):
        self.y_hat = self.model.predict(self.X_test)
        metric = roc_auc_score(self.y_test, self.y_hat)
        return metric

def simulate(model, label):
    CM = ClassyModel(model)
    scores_i = np.zeros((10, 12))
    for i in range(10):
        X, y, vocab = load_data(label, proportion=0.05, min_df=0.01)
        scores = []
        for test_size in [0.98, 0.97, 0.96, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
            CM.split_n_fit(X, y, test_size)
            score = CM.score()
            print("Test size = {}, Class {} = {}, AUC = {}".format(
                len(CM.y_train),
                label,
                sum(CM.y_test),
                score))
            scores.append(score)
        print("----------------------------")
        scores_i[i] = scores
    return np.mean(scores_i, axis=0)
