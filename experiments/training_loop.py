import torch
import numpy as np
import json
import random
import sys

sys.path.append(".")

from core import training as ctr
from core import datasets as cds





def make_deterministic(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(seed)

    # Built-in Python
    random.seed(seed)

def main(params):

    make_deterministic()
    trainer = ctr.DLM_trainer(params['data_directory'])
    trainer.train(1000)



def test_dataset(data_directory, set):
    train_dataset = cds.DLM_dataset(data_directory=data_directory, set=set)
    for i in range(len(train_dataset)):
        _, label = train_dataset.__getitem__(i)
        print("Index: ", i, " Label: ", label)
    print("Test complete")




if __name__ == "__main__":

    with open("config/DL_model_config.json", "r") as fp:
        params = json.load(fp)

    main(params)
