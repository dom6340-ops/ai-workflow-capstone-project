import pytest
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_train_missing_params(client):
    res = client.post('/train')
    assert res.status_code == 202

def test_predict_missing_params(client):
    res =client.get('/predict')
    assert res.status_code == 400

def test_logfile_not_found(client,monkeypatch):
    monkeypatch.setattr('os.path.exists',lambda path: False)
    res = client.get('/logfile?type=predict')
    assert res.status_code == 404