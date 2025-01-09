from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

class Metrics:
    def __init__(self, a, b):
        self.mse = Metrics.mse(a, b)
        self.scc = Metrics.scc(a, b)
    def mse(a, b):
        mse = mean_squared_error(a,b)
        return mse
    def scc(a, b):
        scc, _ = spearmanr(a, b)
        return scc