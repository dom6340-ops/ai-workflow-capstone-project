import os
import traceback
from flask import Flask, request, jsonify

from src.model_files.train import train_all_models
from src.model_files.predict import predict_all_models

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

ALLOWED_DATA_DIRS = ['cs-train','cs-production']

@app.route('/train',methods=['POST'])
def train_endpoint():
    try:
        payload = request.get_json(force=True,silent=True) or {}
        data_dir = payload.get('data_dir','cs-train')
        test = bool(payload.get('test',False))

        if data_dir not in ALLOWED_DATA_DIRS:
            return jsonify({'Status': 'Error', 'Message': f'Invalid data_dir: {data_dir}'}), 400


        train_all_models(data_dir,test)
        return jsonify({'Status':'Accepted','Message':f'Training has begun, with test = {test}',"Data Directory":data_dir}), 202
    except Exception as e:
        trb = traceback.format_exc()
        return jsonify({'Status':"Error","Message":str(e),"Traceback":trb}), 500
    
@app.route('/predict',methods=['GET'])
def predict_endpoint():
    try:
        country = request.args.get('country','all')
        date = request.args.get('date')
        test = bool(request.args.get('test',False))
        prod = bool(request.args.get('prod',False))
        if date is None:
            return jsonify({'Status':'Error','Message': "A valid date is required for query."}), 400
        pred = predict_all_models(country,date,test=test,prod=prod)
        y_rf = pred.get('y_pred_rf')
        y_gbt = pred.get('y_pred_gbt')
        y_true = pred.get('y_true')
        return jsonify({
            'Status': "OK",
            'Country': country,
            'Date': date,
            'y random forest':y_rf,
            'y gradient boosted trees': y_gbt,
            'y true': y_true

        }), 200

    except Exception as e:
        trb = traceback.format_exc()
        return jsonify({"Status":"Error",'Message':str(e),"Traceback":trb}), 500
    
ALLOWED_LOG_TYPES = ['train','predict']

@app.route('/logfiles',methods=['GET'])
def logfiles_endpoint():
        log_type = request.args.get('type')
        log_dir = 'logs'

        if log_type not in ALLOWED_LOG_TYPES:
            return jsonify({'Status': 'Error', 'Message': f'Invalid log type: {log_type}'}), 400
        
        file_path = {'train': os.path.join(log_dir,'train_log.json'),
                        'predict':os.path.join(log_dir,'predict_log.json')}
        path = file_path.get(log_type)
        if not os.path.exists(path):
            return jsonify({'Status': 'Error', 'Message': f'Log file not found. Path: {path}'}), 400

        try:
            with open(path,'r',encoding='utf-8') as f:
                log = f.read()
            return jsonify({'Status':'OK','Path':path,'Log':log}), 200
        
        except Exception as e:
            trb = traceback.format_exc()
            return jsonify({"Status":"Error",'Message':str(e),"Traceback":trb}), 500