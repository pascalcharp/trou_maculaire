from core import datasets as cds
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


mydata = cds.sham_dataset(cardinal=32, image_directory="/Users/pascalcharpentier/PycharmProjects/trou_maculaire_regression_logistique/data/sham_data/")
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(mydata), size=(1,)).item()
    img, label = mydata[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(str(label.item()))
    plt.axis("off")
    tensor_to_PIL = transforms.ToPILImage()
    image = tensor_to_PIL(img)
    plt.imshow(image, cmap="gray")
    plt.show()
