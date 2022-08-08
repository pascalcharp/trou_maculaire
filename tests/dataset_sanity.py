from core import datasets as cds
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


mydata = cds.DLM_dataset(data_directory="/Users/pascalcharpentier/PycharmProjects/trou_maculaire_regression_logistique/data", direction="both", set="test")
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
for i in [1, 2, 13, 14]:
    sample_idx = i-1
    img, label = mydata[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(str(label.item()))
    plt.axis("off")
    tensor_to_PIL = transforms.ToPILImage()
    image = tensor_to_PIL(img)
    plt.imshow(image, cmap="gray")
    plt.show()
