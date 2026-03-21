import os 
import json
from datetime import datetime

LOG_DIR = 'logs'
TRAIN_LOG = os.path.join(LOG_DIR, 'train_log.json')
PREDICT_LOG = os.path.join(LOG_DIR, 'predict_log.json')

def check_log_dir():
    '''
    Checks to see if a the log directory exsists if it does not exist, one is created.

    '''
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def update_train_log(tag, dates, metrics, runtime,model_version, note='', test=False):
    '''
    Updates the train log with various details about the specific model, these details are outlined by the argument descriptions.

    Args:
        tag (str): The tag of the trained model (all or the specific country).
        dates (list): A list of the range of dates the model was trained on.
        metrics (dict): Provides the metrics of the model training such as MAE and/or RMSE.
        runtime (datetime): Datetime of how long the model took to train.
        model_version (float): The model version of the model that was trained.
        note (str): Any notes about the model, typically used to record the type of model such as random forest.
        test (bool): Indicates whether the fucntion is being used for a test or not. 
        
    '''
    check_log_dir()
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'tag': tag,
        'dates': dates,
        'model_version': model_version,
        'runtime': runtime,
        'metrics': metrics,
        'test': test
    }
    
    if os.path.exists(TRAIN_LOG):
        try:
            with open(TRAIN_LOG, 'r') as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            print("Warning: log file was corrupt, starting fresh.")
            log_data = []
    else:
        log_data = []
    
    log_data.append(log_entry)
    
    with open(TRAIN_LOG, 'w') as f:
        json.dump(log_data, f, indent=4)
    if test:
        return TRAIN_LOG

def update_predict_log(tag, dates, predictions, runtime, model_version, note='', test=False):
    '''
    Trains all models on the training data for each country. A random forest model and gradient boosted tree model is train on each country and all the countries within the training data set.
    The models are then saved to the Models directory. 

    Args:
        tag (str): The tag of the model used for predictions (all or the specific country).
        dates (str): The date the model was predicting.
        predictions (float): The predicted value produced by the model or models.  
        runtime (datetime): Datetime of how long the model took to train.
        model_version (float): The model version of the model that was trained.
        note (str): Any notes about the model, typically used to record the type of model such as random forest.
        test (bool): Indicates whether the fucntion is being used for a test or not. 
        
    '''
    check_log_dir()
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'tag': tag,
        'dates': dates,
        'model_version': model_version,
        'runtime':runtime,
        'predictions': predictions,
        'notes': note,
        'test': test
    }
    
    if os.path.exists(PREDICT_LOG):
        with open(PREDICT_LOG, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    log_data.append(log_entry)
    
    with open(PREDICT_LOG, 'w') as f:
        json.dump(log_data, f, indent=4)
    if test:
        return PREDICT_LOG