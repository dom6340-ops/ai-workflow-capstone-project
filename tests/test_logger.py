import os
import json
import pytest
from src.logger import update_predict_log,update_train_log

def test_update_train_log(tmp_path,monkeypatch):
    monkeypatch.setattr('src.logger.LOG_DIR',str(tmp_path / 'logs'))
    os.chdir(str(tmp_path))
    country='all'
    date = '2019-08-01'
    metric = [123]
    runtime = "00:00:01"
    model_version = 0.1

    path = update_train_log(country,date,metric, runtime,model_version,test=True)

    assert os.path.exists(path), f"Expected log file at {path} but it was not created"

    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)

    assert isinstance(data,list)

    last = data[-1]

    assert last['test']
    assert last['tag'] == country
    assert last['dates'] == date
    assert last['metrics'] == metric
    assert last['runtime'] == runtime
    assert last['model_version'] == model_version


def test_update_predict_log(tmp_path,monkeypatch):

    monkeypatch.setattr('src.logger.LOG_DIR',str(tmp_path / 'logs'))
    os.chdir(str(tmp_path))
    country='all'
    date = '2019-08-01'
    predictions = [123]
    runtime = "00:00:01"
    model_version = 0.1

    path = update_predict_log(country,date,predictions, runtime, model_version,test=True)

    assert os.path.exists(path), f"Expected log file at {path} but it was not created"

    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)

    assert isinstance(data,list)

    last = data[-1]

    assert last['test']
    assert last['tag'] == country
    assert last['dates'] == date
    assert last['predictions'] == predictions
    assert last['runtime'] == runtime
    assert last['model_version'] == model_version