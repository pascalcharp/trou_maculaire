import sklearn.metrics
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
import random
from sklearn import metrics

from torch.utils.data import DataLoader




normalisation_factors_means = [0.163549, 0.163544, 0.163547]
normalisation_factors_std = [0.133186, 0.133183, 0.133186]




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




class DLM_CBR_tiny(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = 0.0
        self.feature_size = 512

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
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
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
        # ts_accuracy = self.accuracy(y_hat, y)
        # ts_auroc = self.auroc(y_hat, y)
        self.log("train_loss", ts_loss, on_epoch=True, logger=True)
        # self.log("train_accuracy", ts_accuracy, on_epoch=True, logger=True)
        # self.log("train_auroc", ts_auroc, on_epoch=True, logger=True)
        return ts_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        vs_loss = self.loss(torch.squeeze(y_hat), y)
        # vs_accuracy = self.accuracy(y_hat, y)
        # vs_auroc = self.auroc(y_hat, y)
        self.log("validation_loss", vs_loss, on_epoch=True, logger=True)
        # self.log("validation_accuracy", vs_accuracy, on_epoch=True, logger=True)
        # self.log("validation_auroc", vs_auroc, on_epoch=True, logger=True)
        return vs_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model.forward(X)
        ts_loss = self.loss(torch.squeeze(y_hat), y)
        # ts_accuracy = self.accuracy(y_hat, y)
        # ts_auroc = self.auroc(y_hat, y)
        self.log("test_loss", ts_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_accuracy", ts_accuracy, on_epoch=True, logger=True)
        # self.log("test_auroc", ts_auroc, on_epoch=True, logger=True)
        return ts_loss


class DLM_trainer:
    def __init__(self, directory):
        self.save_model_path = "saved_models/model"

        self.model = DLM_CBR_tiny()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.loss = nn.MSELoss()
        self.validation_loss_target = 0.01

        self.train_batch_size = 32
        self.validation_batch_size = 1
        self.test_batch_size = 1

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-4)

        self.training_dataset = DLM_dataset(directory, set="train", direction="both")
        self.validation_H_dataset = DLM_dataset(directory, set = "val", direction="H")
        self.validation_V_dataset = DLM_dataset(directory, set="val", direction="V")
        self.test_dataset = DLM_dataset(directory, set="test", direction="both")

        self.train_loader = DataLoader(self.training_dataset, batch_size=32, num_workers=6, shuffle=True)
        self.validation_H_loader = DataLoader(self.validation_H_dataset, batch_size=21, num_workers=6, shuffle=True)
        self.validation_V_loader = DataLoader(self.validation_V_dataset, batch_size=21, num_workers=6, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=34, num_workers=6, shuffle=True)

    def train(self, epochs=500):

        max_auroc = 0.0
        best_model = {}

        for epoch in range(epochs):

            self.model.train()
            training_loss = 0.0

            for X, y in self.train_loader:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                self.optimizer.zero_grad()
                logits = self.model(X)
                pred_probab = torch.sigmoid(logits)
                loss = self.loss(pred_probab, y)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()



            training_loss = training_loss / len(self.train_loader)

            if epoch % 50 == 49:
                validation_loss = 0.0
                validation_auroc = 0.0



                validation_F1 = 0.0
                validation_accuracy = 0.0

                print (f"Epoch {epoch} : validation")
                self.model.eval()
                with torch.no_grad():

                    V_loss, V_labels, V_probabilities = self.perform_inference_on(self.validation_V_loader)
                    H_loss, H_labels, H_probabilities = self.perform_inference_on(self.validation_H_loader)

                assert(np.array_equal(V_labels, H_labels))
                probabilities = 0.5 * (V_probabilities + H_probabilities)



                auroc = sklearn.metrics.roc_auc_score(V_labels, probabilities)
                # F1 = sklearn.metrics.f1_score(labels, probabilities)
                # accuracy = sklearn.metrics.accuracy_score(labels, probabilities)
                # fpr, tpr, thr = metrics.roc_curve(y_true=labels, y_score=probabilities, pos_label=1.0)


                validation_auroc += auroc
                # validation_F1 += F1
                validation_loss += 0.5 * (V_loss + H_loss)
                # validation_accuracy += accuracy



                validation_loss = validation_loss / len(self.validation_V_loader)
                validation_auroc = validation_auroc / len(self.validation_V_loader)
                if (validation_auroc > max_auroc):
                    max_auroc = validation_auroc
                    best_model = self.model.state_dict()

                # validation_F1 = validation_F1 / len(self.validation_loader)
                # validation_accuracy = validation_accuracy / len(self.validation_loader)

                print(f"Epoch {epoch} $ Training loss $ {training_loss} $ Validation loss $ {validation_loss} $ Validation auroc $ {validation_auroc}") # $ Validation acuuracy $ {validation_accuracy} $ Validation F1 $ {validation_F1}")

            else:
                print(f"Epoch {epoch} $ Training loss $ {training_loss}")

        print("Meilleur AUROC obtenu: ", max_auroc)
        print("Sauvegarde du modèle")
        torch.save(best_model, self.save_model_path)

    def perform_inference_on(self, dataloader):

        for X, y in dataloader:

            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            logits = self.model(X)
            pred_probab = torch.sigmoid(logits)
            loss = self.loss(pred_probab, y)

            labels = y.cpu().numpy()
            probabilities = pred_probab.cpu().numpy()

            return loss.item(), labels, probabilities


# def main(data_directory, train_dataset_batch_size, enable_progress_bar):
#     # Données d'entraînement
#     train_dataset = DLM_dataset(data_directory=data_directory, set="train")
#     train_loader = DataLoader(train_dataset, batch_size=train_dataset_batch_size, num_workers=4)
#
#     # Données de validation
#     val_dataset = DLM_dataset(data_directory=data_directory, set="val")
#     val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
#
#     # Données de test
#     test_dataset = DLM_dataset(data_directory=data_directory, set="test")
#     test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
#
#     # Modèle de deep learning et module d'entraînement
#     CBR_Tiny = DLM_module(model=DLM_CBR_tiny)
#     trainer = pl.Trainer(enable_progress_bar=enable_progress_bar, log_every_n_steps=6, flush_logs_every_n_steps=6, max_epochs=1000, accelerator='gpu', devices=1)
#
#     # Entraînement
#     trainer.fit(model=CBR_Tiny, train_dataloaders=train_loader, val_dataloaders=val_loader)
#
#     # Test
#     trainer.test(model=CBR_Tiny, dataloaders=test_loader)
#     # trainer.test(model=CBR_Tiny, dataloaders=[train_loader, val_loader])


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
    trainer = DLM_trainer(params['data_directory'])
    trainer.train(1000)



def test_dataset(data_directory, set):
    train_dataset = DLM_dataset(data_directory=data_directory, set=set)
    for i in range(len(train_dataset)):
        _, label = train_dataset.__getitem__(i)
        print("Index: ", i, " Label: ", label)
    print("Test complete")




if __name__ == "__main__":

    with open("DL_model_config.json", "r") as fp:
        params = json.load(fp)

    main(params)


