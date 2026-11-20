import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from phishing_ml.preprocess import make_pipeline

def test_preprocessor_train_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    test = pd.DataFrame({"a": [100.0, 200.0], "b": [5.0, 5.0]})
    pipe = make_pipeline(LogisticRegression(max_iter=1000), True)
    pipe.fit(train, [0, 1, 0])
    scaler = pipe.named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    # transforming test must not change fitted params
    pipe.predict(test)
    assert np.allclose(scaler.mean_, mean_before)
    # median imputer fitted on train only
    imp = pipe.named_steps["imputer"]
    assert imp.statistics_ is not None
