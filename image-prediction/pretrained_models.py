# Import the torchvision.models module
import torchvision.models as models

# Get the names of all the models in the module
model_names = [name for name in dir(models) if name[0].isupper()]

# Print the model names
print("The models from torchvision are:")
for name in model_names:
    print(name)
