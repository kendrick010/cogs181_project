import os
import opendatasets as od

DATASET = 'https://www.kaggle.com/datasets/chandlertimm/unified'

def get_dataset():
    os.chdir('./dataset')

    # Skip download if dataset exists
    if not(os.path.isdir('unified')):
        od.download(DATASET)

if __name__ == "__main__":
    get_dataset()