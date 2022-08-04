import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



normalisation_factors_means = [0.163549, 0.163544, 0.163547]
normalisation_factors_std = [0.133186, 0.133183, 0.133186]




class sham_dataset(torch.utils.data.dataset.Dataset):
    pass



class DLM_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_directory="/home/chapas/trou_maculaire/data", set="train", direction="both"):

        input_filename = data_directory + "/clinical_data_" + set + ".csv"
        self.data_directory = data_directory
        self.image_directory = data_directory + "/oct_data/" + set + "/octs/"
        self.set = set

        if direction.upper() not in ["BOTH", "H", "V"]:
            raise "Invalid direction option in dataset"
        self.direction = direction.upper()

        self.data = pd.read_csv(input_filename)

        # Les labels sont stockés dans une liste de d'enregistrements.  Chaque enregistrement contient un champ 'id'
        # qui est le numéro du patient, et correspond au numéro du fichier .tiff correspondant, et un champ 'responder'
        # qui est True ou False et décrit la réponse clinique, qui est notre variable dépendante.
        self.get_labels_from_dataframe()
        self.labels = self.data.to_dict('records')


    def __len__(self):

        # 2 car chaque label dans self.labels correspond à 2 scans: le scan horizontal et le scan vertical.
        if self.direction == "BOTH":
            return 2 * len(self.labels)
        else:
            return len(self.labels)

    def __getitem__(self, index):

        if self.direction == "BOTH":
            # Il y a deux scans par patient
            # index pair: scan horizontal, index impair: scan vertical
            if index % 2 == 0:
                patient_idx = index // 2
                oct_direction = "H"
            else:
                patient_idx = (index - 1) // 2
                oct_direction = "V"
        else:
            patient_idx = index
            oct_direction = self.direction

        # Retrouver les informations du patient
        record = self.labels[patient_idx]
        label = torch.as_tensor(record['responder']).float().unsqueeze(0)
        image_file_name = self.image_directory + str(record['id']) + "_baseline_" + oct_direction + ".tiff"

        try:

            with Image.open(image_file_name) as image:
                rgb_image = image.convert("RGB")

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
                tensor = first_transformation(rgb_image)

            # Normalisation des 3 canaux

            final_transformation = transforms.Compose([transforms.Normalize(normalisation_factors_means, normalisation_factors_std, inplace=True)])
            final_transformation(tensor)

            return  tensor, label

        except IOError:
            print(f"Fichier image ne peut être récupéré: {image_file_name}")
            exit(1)




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


