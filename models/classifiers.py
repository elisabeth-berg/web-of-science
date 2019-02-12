from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from models.loader import *


class ClassyModel(object):

    def __init__(self, model):
        self.model = model

    def split_n_fit(self, X, y, test_size, train_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, train_size=train_size, random_state=random_state)

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_hat = self.model.predict(self.X_test)

    def score(self):
        self.y_hat = self.model.predict(self.X_test)
        metric = roc_auc_score(self.y_test, self.y_hat)
        return metric

def simulate(model, label, target_p, train_sizes, n, random_state):
    """
    We run this n times.
    The randomness comes from the load_data function: in particular, which
    subset of the positive class is used. We keep this proportion at target_p, but
    randomize the sample that is chosen. This is meant to simulate the
    differences in quality of data that might occur.

    The mean of the result (scores_i) should give an idea of the (average) rate
    at which performance improves.
    """
    CM = ClassyModel(model)
    scores_i = np.zeros((n, len(train_sizes)))
    
    # Keep the size of the testing set constant
    test_size = 1 - max(train_sizes)
    
    for i in range(n):
        X, y, vocab = load_data(label=label, target_p=target_p, min_df=0.005)
        scores = []
        for train_size in train_sizes:
            CM.split_n_fit(X, y, test_size, train_size, random_state)
            score = CM.score()
#            print("Test size = {}, Class {} = {}, AUC = {}".format(
#                len(CM.y_train),
#                label,
#                sum(CM.y_test),
#                score))
            scores.append(score)
#        print("----------------------------")
        scores_i[i] = scores
    return scores_i, X, y, vocab
