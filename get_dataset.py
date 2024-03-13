import os
import opendatasets as od

KAGGLE = 'kaggle.json'
DATASET = 'https://www.kaggle.com/datasets/chandlertimm/unified'

def get_dataset():
    od.download(DATASET)

if __name__ == "__main__":
    os.chdir('./dataset')

    if os.path.isfile(KAGGLE):
        get_dataset()

    else: print('kaggle.json not found in /dataset directory')