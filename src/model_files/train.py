import os
import time
import sys
import json
import numpy as np
import ydf
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from src.logger import update_train_log
from src.ingest_data import ingest_data
from solution_guidance.cslib import engineer_features
from solution_guidance.model import _model_train

MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE= "Gradient Boosted Tree Model"

def train_all_models(dir='cs-train', test=False):
    '''
    This function trains both a gradient boosted trees model and the random forest model provided from the solution guidance. A model of each type is trained
    for each country in the training data as well as the entire training data set. The models are then saved in the models directory where they can be loaded for predictions.
    The training log is updated upon commpletion of training each model.

    Args:
        dir (str): The name or path of the directory containing the training data within the data directory.
        test (bool): A bool that indicated whether the function is being used for a test or not.
    '''

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
    else:
        dataset = ingest_data(train=dir)
        df_train = dataset['train']
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    for country, df in df_train.items():
        
        time_start = time.time()
        if test and country not in ['all','united_kingdom']:
            continue

        _model_train(df,country,test=test)

        if not test:
            X, y, dates = engineer_features(df)
            
        X['revenue'] = y
        holdout = int(len(X)*0.8)
        df_train,df_holdout = X[:holdout],X[holdout:]

        learner = ydf.GradientBoostedTreesLearner(
            label='revenue',
            task=ydf.Task.REGRESSION,
            num_trees=300,
            max_depth=6,
            shrinkage=0.05,
            subsample=0.8,
            use_hessian_gain=True,
            validation_ratio=0.15
        )
        gbt_model =learner.train(df_train)
        preds = gbt_model.predict(df_holdout.drop(columns=['revenue']))
        rev_true = df_holdout['revenue'].values

        mae = float(np.mean(np.abs(preds-rev_true)))
        rmse = float(np.sqrt(np.mean((preds-rev_true)**2)))
        eval_metrics = {'mae':mae,'rmse':rmse}

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        model_path = os.path.join(MODEL_DIR, f'{country}_gbt_{MODEL_VERSION}')
        gbt_model.save(model_path)

        m, s = divmod(time.time() - time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d" % (h, m, s)

        update_train_log(country, (str(dates[0]), str(dates[-1])),
                         eval_metrics, runtime, MODEL_VERSION,
                         MODEL_VERSION_NOTE, test=test) 


if __name__ == "__main__":
    train_all_models(test=False) 
    print("Model training completed.")
