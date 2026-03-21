import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from predict import predict_all_models

def evaluate_models(y_true,rf_preds,gbt_preds,dir='model_reports'):
    '''
    This function plots the predictions of both the gradient boosted trees model, random forest model, and true revenue values against eachother for comparison.

    Args: 
        y_true (list): List of true values.
        rf_preds (list): List of predicition values from the random forest model.
        gbt_preds (list): List of prediction values from the gradient boosted trees model.

    Returns:
        A dictionary of metrics for both models.
        mae_rf (float): The mean absolute error of the random forest model.
        mae_gbt (float): The mean absolute error of the gradient boosted trees model.
        rmse_rf (float): The root mean squared error of the random forest model.
        rmse_gbt (float): The root mean squared error of the gradient boosted trees model.
    '''

    if not os.path.exists(dir):
        os.mkdir(dir)
    rf_mae = mean_absolute_error(y_true,rf_preds)
    gbt_mae = mean_absolute_error(y_true,gbt_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_true,rf_preds))
    gbt_rmse = np.sqrt(mean_squared_error(y_true,gbt_preds))

    plt.figure(figsize=(12,8))
    plt.plot(y_true,label='True Values',color='black')
    plt.plot(rf_preds,label='RF Predictions',color='blue')
    plt.plot(gbt_preds,label='GBT Predictions',color='red')
    plt.legend()
    plt.title("Model Comparison")
    plt.xlabel("Time")
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig(f'{dir}/model_comparison_report.png')

    return({
        'mae_rf': rf_mae,
        'mae_gbt':gbt_mae,
        'rmse_rf':rf_rmse,
        'rmse_gbt':gbt_rmse
    })

if __name__ == '__main__':
    y_true, y_rf,y_dense = [],[],[]
    for i in range(1,31):
        y = predict_all_models(target_date=f"2019-10-{i:02d}",prod=False)
        y_true.append(y['y_true'])
        y_rf.append(y['y_pred_rf'])
        y_dense.append(y['y_pred_gbt'])    
    metrics = evaluate_models(y_true,y_rf,y_dense)
    print(metrics)