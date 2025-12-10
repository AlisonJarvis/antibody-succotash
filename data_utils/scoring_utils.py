import scipy as sp

def score_spearman(model, X, y_true):
    y_pred = model.predict(X)
    return sp.stats.spearmanr(y_true, y_pred.reshape(-1)).statistic
