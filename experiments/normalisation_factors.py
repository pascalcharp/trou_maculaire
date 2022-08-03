import torch
import pandas as pd
import json
import os
from PIL import Image
import torchvision.transforms as transforms

with open("../config/DL_model_config.json", "r") as fp:
    params = json.load(fp)

image_directory = params["data_directory"] + "/oct_data/" + params["set"] + "/octs"

result_means = []

for filename in os.listdir(image_directory):
    f = os.path.join(image_directory, filename)
    if os.path.isfile(f):
        record = {}
        print(f)
        image = Image.open(f).convert("RGB")
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.3, hue=0.3),
            transforms.RandomAffine(0, translate=(0.2, 0.05)),
            transforms.ToTensor()
        ]
        first_transformation = transforms.Compose(transform_list)
        tensor = first_transformation(image)

        mean_tensor = torch.mean(tensor, (1, 2))
        mean_std = torch.std(tensor, (1, 2))

        record['mean_channel_0'] = mean_tensor[0].item()
        record['mean_channel_1'] = mean_tensor[1].item()
        record['mean_channel_2'] = mean_tensor[2].item()
        record['std_channel_0'] = mean_std[0].item()
        record['std_channel_1'] = mean_std[1].item()
        record['std_channel_2'] = mean_std[2].item()

        result_means.append(record)


df = pd.DataFrame.from_records(result_means)

print(df.describe())

