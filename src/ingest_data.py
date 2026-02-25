import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution_guidance.cslib import fetch_ts

def ingest_data(train='cs-train', test='cs-production'):
    '''
    Ingests the training and test datasets from the specified directories.

    Args:
        train (str): Directory path for the training dataset.
        test (str): Directory path for the test dataset.

    Returns:
        dict: A dictionary containing the ingested datasets with keys 'train' and 'test'.
    '''

    datasets = {}
    for label,dir in [('train',train),('test',test)]:
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Data directory '{dir}' not found.")
        datasets[label] = fetch_ts(dir,clean=False)

    return datasets

if __name__ == "__main__":
    datasets = ingest_data()
    print(f'Loaded the following datasets: {datasets.keys()}')
    