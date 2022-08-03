import sklearn.metrics
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader



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
        self.validation_H_loader = DataLoader(self.validation_H_dataset, batch_size=21, num_workers=6, shuffle=False)
        self.validation_V_loader = DataLoader(self.validation_V_dataset, batch_size=21, num_workers=6, shuffle=False)
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

