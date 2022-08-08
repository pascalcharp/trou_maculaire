from core import datasets as cds
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


mydata = cds.DLM_dataset(data_directory="/Users/pascalcharpentier/PycharmProjects/trou_maculaire_regression_logistique/data", direction="both", set="test")
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
listindex = [0, 1, 12, 13]
for i in range(4):
    sample_idx = listindex[i]
    img, label = mydata[sample_idx]
    figure.add_subplot(rows, cols, i+1)
    plt.title(str(label.item()))
    plt.axis("off")
    tensor_to_PIL = transforms.ToPILImage()
    image = tensor_to_PIL(img)
    plt.imshow(image, cmap="gray")
    plt.show()
