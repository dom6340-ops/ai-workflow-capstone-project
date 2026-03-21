import os
import pytest
from src.model_files.train import train_all_models

def test_train_model_creates_model(tmp_path,monkeypatch):
    cwd = os.getcwd()
    try:
        monkeypatch.setattr('src.model_files.train.MODEL_DIR',str(tmp_path / 'models'))
        monkeypatch.setattr('src.model_files.train.MODEL_VERSION', '0.1')

        os.chdir(str(tmp_path))
        train_all_models('cs-train')

        assert (tmp_path / 'models').exists()

    finally:
        os.chdir(cwd)
