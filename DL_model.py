import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torchmetrics
from PIL import Image
import torchvision.transforms as transforms
import json

from torch.utils.data import DataLoader




normalisation_factors_means = [0.163549, 0.163544, 0.163547]
normalisation_factors_std = [0.133186, 0.133183, 0.133186]




class DLM_dataset(torch.utils.data.dataset.Dataset):
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
        label = torch.as_tensor(float(record['responder']))
        image_file_name = self.image_directory + str(record['id']) + "_baseline_" + oct_direction + ".tiff"
        # print("Fetching file: ", image_file_name)
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

        return  tensor, label

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




class DLM_CBR_tiny(nn.Module):

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
            nn.Linear(self.feature_size, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return self.head(x)




class DLM_module(pl.LightningModule):
    def __init__(self, model):
        super(DLM_module, self).__init__()
        self.model=model()
        self.loss = nn.MSELoss()
        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC()

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        ts_loss = self.loss(torch.squeeze(y_hat), y)
        ts_accuracy = self.accuracy(y_hat, y)
        ts_auroc = self.auroc(y_hat, y)
        self.log("train_loss", ts_loss, on_epoch=True, logger=True)
        # self.log("train_accuracy", ts_accuracy, on_epoch=True, logger=True)
        self.log("train_auroc", ts_auroc, on_epoch=True, logger=True)
        return ts_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        vs_loss = self.loss(torch.squeeze(y_hat), y)
        # vs_accuracy = self.accuracy(y_hat, y)
        vs_auroc = self.auroc(y_hat, y)
        self.log("validation_loss", vs_loss, on_epoch=True, logger=True)
        self.log("validation_accuracy", vs_accuracy, on_epoch=True, logger=True)
        self.log("validation_auroc", vs_auroc, on_epoch=True, logger=True)
        return vs_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        ts_loss = self.loss(torch.squeeze(y_hat), y)
        # ts_accuracy = self.accuracy(y_hat, y)
        ts_auroc = self.auroc(y_hat, y)
        self.log("test_loss", ts_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", ts_accuracy, on_epoch=True, logger=True)
        self.log("test_auroc", ts_auroc, on_epoch=True, logger=True)
        return ts_loss




def main(data_directory, train_dataset_batch_size, enable_progress_bar):
    # Données d'entraînement
    train_dataset = DLM_dataset(data_directory=data_directory, set="train")
    train_loader = DataLoader(train_dataset, batch_size=train_dataset_batch_size, num_workers=4)

    # Données de validation
    val_dataset = DLM_dataset(data_directory=data_directory, set="val")
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)

    # Données de test
    test_dataset = DLM_dataset(data_directory=data_directory, set="test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)

    # Modèle de deep learning et module d'entraînement
    CBR_Tiny = DLM_module(model=DLM_CBR_tiny)
    trainer = pl.Trainer(enable_progress_bar=enable_progress_bar, log_every_n_steps=6, flush_logs_every_n_steps=6, max_epochs=1000, accelerator='gpu', devices=1)

    # Entraînement
    trainer.fit(model=CBR_Tiny, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    trainer.test(model=CBR_Tiny, dataloaders=test_loader)
    # trainer.test(model=CBR_Tiny, dataloaders=[train_loader, val_loader])


def test_dataset(data_directory, set):
    train_dataset = DLM_dataset(data_directory=data_directory, set=set)
    for i in range(len(train_dataset)):
        _, label = train_dataset.__getitem__(i)
        print("Index: ", i, " Label: ", label)
    print("Test complete")




if __name__ == "__main__":

    with open("DL_model_config.json", "r") as fp:
        params = json.load(fp)

    data_directory = params['data_directory']
    train_dataset_batch_size = params['train_dataset_batch_size']
    enable_progress_bar = params['enable_progress_bar']
    main(data_directory=data_directory, train_dataset_batch_size=train_dataset_batch_size, enable_progress_bar=enable_progress_bar)

    # test_dataset(data_directory=data_directory, set="test")
