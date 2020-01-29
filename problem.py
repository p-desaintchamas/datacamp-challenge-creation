import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit


problem_title = 'Prediction of daily validation in Paris public underground transports'
_target_column_name = 'NB_VALD'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow

class VALID(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor']):  #, 'award_notices_RAMP.csv']):
        super(VALID, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = VALID()

# define the score (specific score for the VALID problem)
class VALID_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='ratp error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        loss = np.sum(np.log(np.cosh(y_pred - y_true)))
        return loss

score_types = [
    VALID_error(name='ratp error', precision=2),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X['LIBELLE_ARRET'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,
                       compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='./data/'):
    f_name = 'ratp_validation_TRAIN.csv.zip'
    return _read_data(path, f_name)

def get_test_data(path='./data/'):
    f_name = 'ratp_validation_TEST.csv.zip'
    return _read_data(path, f_name)
