import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json

normalisation_factors_means = [0.163549, 0.163544, 0.163547]
normalisation_factors_std = [0.133186, 0.133183, 0.133186]

class DL_model_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_directory="/home/chapas/trou_maculaire/data", set="train"):

        input_filename = data_directory + "/clinical_data_" + set + ".csv"
        self.data_directory = data_directory
        self.image_directory = data_directory + "/oct_data/" + set + "/octs/"
        self.set = set

        self.data = pd.read_csv(input_filename)

        # Les labels sont stockés dans une liste de d'enregistrements.  Chaque enregistrement contient un champ 'id'
        # qui est le numéro du patient, et correspond au numéro du fichier .tiff correspondant, et un champ 'responder'
        # qui est True ou False et décrit la réponse clinique, qui est notre variable dépendante.
        self.get_labels_from_dataframe()
        self.labels = self.data.to_dict('records')


    def __len__(self):

        # 2 car chaque label dans self.labels correspond à 2 scans: le scan horizontal et le scan vertical.
        return 2 * len(self.labels)

    def __getitem__(self, index):

        # Il y a deux scans par patient
        # index pair: scan horizontal, index impair: scan vertical
        if index % 2 == 0:
            patient_idx = index // 2
            oct_direction = "H"
        else:
            patient_idx = (index - 1) // 2
            oct_direction = "V"

        # Retrouver les informations du patient
        record = self.labels[patient_idx]
        label = float(record['responder'])
        image_file_name = self.image_directory + str(record['id']) + "_baseline_" + oct_direction + ".tiff"
        image = Image.open(image_file_name).convert("RGB")

        # Augmentation des données.  Ces transformations correspondent au niveau 'medium' dans le programme de
        # Mathieu Godbout
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.3, hue=0.3),
            transforms.RandomAffine(0, translate=(0.2, 0.05)),
            transforms.ToTensor()
        ]
        first_transformation = transforms.Compose(transform_list)
        tensor = first_transformation(image)

        # Normalisation des 3 canaux

        final_transformation = transforms.Compose([transforms.Normalize(normalisation_factors_means, normalisation_factors_std, inplace=True)])
        final_transformation(tensor)

        return label, tensor

    def get_labels_from_dataframe(self):

        # Retirer les colonnes superflues
        self.data.drop(['age', 'sex', 'pseudophakic', 'mh_duration', 'elevated_edge', 'mh_size', 'VA_2weeks', 'VA_3months', 'VA_12months'], inplace=True, axis=1)

        # Patcher les données manquantes
        self.data['VA_baseline'].replace(0, np.nan, inplace=True)
        self.data['VA_baseline'].fillna(self.data['VA_baseline'].mean(), inplace=True)
        self.data.replace(-9, np.nan, inplace=True)
        self.data.fillna(self.data.mean(), inplace=True)

        # Définir la variable dépendante, les "labels": amélioration de plus de 15 sur 6 mois
        self.data['responder'] = (self.data['VA_6months'] - self.data['VA_baseline']) >= 15

        # Éliminer les colonnes restantes: il ne restera que la colonne 'responder'
        self.data.drop(['VA_baseline', 'VA_6months'], inplace=True, axis=1)

class DL_model_CBR_tiny(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = 0.0
        self.feature_size = 256

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_size, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return self.head(x)

if __name__ == "__main__":
    with open("DL_model_config.json", "r") as fp:
        cfg = json.load(fp)



    test_dataset = DL_model_dataset(set=cfg["set"], data_directory=cfg["data_directory"])
    print(test_dataset.labels)

    label, image = test_dataset.__getitem__(12)
    print(label)
    print(image)
    print(image.shape)

    label, image = test_dataset.__getitem__(13)
    print(label)
    print(image)
    print(image.shape)

