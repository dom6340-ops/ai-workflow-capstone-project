import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def load_predict_log(path='logs'):
    '''
    This function loads the prediction logs from the log directory.

    Args:
        path (str): relative path to the log directory.
    '''
    path = os.path.join(path,'predict_log.json')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_pred_trend(date="2018-01-05",outdir='model_reports'):
    '''
    This function plots the predictions for a target date from the prediction log to monitor the predcitions of the model over time and model version.

    Args:
        date (str): The date which is will plot the predictions from the prediction log.
        outdir (str): The output directory the report will store the figure in. 
    '''
    os.makedirs(outdir,exist_ok=True)

    df = load_predict_log()
    df['predicitons'] = df['predictions'].map('{:.2f}'.format)
    if date not in df['dates'].values:
        raise ValueError('Date is not in prediction logs.')
    df['predicitons'] = df['predicitons'].where(df['dates']==date)   
    df['timestamps'] =pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df,x='timestamp',y='predicitons',hue='notes',marker='o')
    plt.title('Target Date Predictions over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Predicted Revenue')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'Target_Date_Predictions_Over_Time.png'))
    print("Target Date Predictions Over Time saved to model_reports directory.")

if __name__ == '__main__': 
    plot_pred_trend()
