import os
import time
import sys
import numpy as np
import ydf
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from src.logger import update_predict_log
from src.ingest_data import ingest_data
from solution_guidance.model import model_predict
from solution_guidance.cslib import engineer_features

def predict_all_models(country="all",target_date="2018-01-05",model_version=0.1,test=False,prod=False):
    '''
    This fucntion uses both the gradient boosted trees model and random forest model of a specific country or all the countries to predict the revenue of a certain date. 
    The function also logs the predictions within the prediction log file.

    Args:
        country (str): Indicates whether to use a specific country's model or the 'all' model.
        target_date (str): Indicates the target date for predictions in a "year-mm-dd" format. 
        model_version (float): Indicates which model version to use for the predictions. 
        test (bool): Indicates whether the function is being used in a test or not. 
        prod (bool): Indicates whether the target_date is within the production data rather than the test data. 

    Returns:
        A dictionary of the predctions
        y_pred_rf (float): The prediction determined by the random forest model.
        y_pred_gbt (float): The prediction determined by the gradient boosted trees model.
        y_true (dloat): The true value. 
    '''
    timer_start = time.time()
    date = target_date.split('-')
    if len(date) != 3:
        raise ValueError('Invalid target date. Please use a "year-month-day" format.')
    rf_result= model_predict(country,date[0],date[1],date[2],test=test,prod=prod)
    gbt_model = ydf.load_model(f'models\\{country}_gbt_{model_version}')
    if prod:
        ts_data = ingest_data()['test']
    else:
        ts_data = ingest_data()['train']
    
    all_data = {}
    
    df = ts_data[country]
    X,y,dates = engineer_features(df,training=False)
    dates = np.array([str(d) for d in dates])
    all_data[country] = {"X":X,"y":y,"dates": dates}

    if target_date not in all_data[country]['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,all_data['dates'][0],all_data['dates'][-1]))
    
    date_index= np.where(all_data[country]['dates'] == target_date)[0][0]
    query= all_data[country]['X'].iloc[[date_index]]
    
    pred = gbt_model.predict(query)[0].astype(float)

    m,s = divmod((time.time()-timer_start),60)
    h,m = divmod(m,60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    update_predict_log(country,target_date,pred,runtime,model_version,note='gbt')

    return ({'y_pred_rf':rf_result['y_pred'],'y_pred_gbt':pred,'y_true':y[date_index]})

if __name__ == '__main__':
        predict_all_models()