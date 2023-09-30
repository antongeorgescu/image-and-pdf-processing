import torchvision.datasets as datasets
cifar10 = datasets.CIFAR10(root='./data-samples/images/cifar-10', train=True, download=True)
print(cifar10.targets[:10])

PICTURE = "pic01.jpg" # cat
# PICTURE = "pic02.jpg" # dog

# classes
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(cifar10.classes[:10])

import torch
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

import os
from PIL import Image
from torchvision import transforms

# Loop through the files in the folder
for file in os.listdir('./pictures/classify'):
    input_image = Image.open(f'./pictures/classify/{file}')
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    input_tensor = preprocess(input_image)

    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0) # add batch dimension
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    # switch to evaluation mode and disable gradients
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    # get the predicted class index or probability
    _, pred = torch.max(output, 1) # index

    probabilities = torch.softmax(output, 1) # probability
    print(f"For image {file} prediction is [{cifar10.classes[pred[0].item()]}] with probability {'{:.2%}'.format(probabilities[0][pred[0].item()].item())}")

print("Done.")