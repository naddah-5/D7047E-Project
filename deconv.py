import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets

from model.model import CNN
from model.deconv_model import DeconvCNN
from application.visualize_deconv import visualize
from application.params import update_params
import ast

from PIL import Image


my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
#my_device = torch.device("cpu")
print("using device ", my_device)

model = CNN(class_count=2, device=my_device)
model.to(my_device)
model.load_state_dict(torch.load('best_network.pt'))

deconv_model = DeconvCNN(class_count=2, device=my_device)
deconv_model.to(my_device)

# new_params = {'scale': [512, 512]}
# update_params(new_params)



#loads all params
params = {}
with open('best_network_params.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                params[key] = ast.literal_eval(value)



scale=params['scale']


transform = transforms.Compose([
    # transforms.RandomApply(
    #     transforms.RandomRotation(enumerate(range(1,359))),
    #     p=0.01
    # ),
    # transforms.RandomInvert(p=0.5),
    # transforms.RandomPerspective(distortion_scale=0.1, p=0.01),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomHorizontalFlip(p=0.1),
    # transforms.RandAugment(),
    transforms.Resize(scale),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_transform = transforms.Compose([
    transforms.Resize(scale),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


image = Image.open('dataset/temp/chest_xray_both/train/PNEUMONIA/0a03fbf6-3c9a-4e2e-89ce-c7629ae43a27.jpg')
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(my_device)
#print(image_tensor.shape)

predictions, our_features, pool_indices, sizes  = model.forward(image_tensor)

deconv_output = deconv_model.forward(our_features, pool_indices, sizes=sizes)

image_tensor = image_tensor[0]
print('predicted:\n',predictions)
visualize(image_tensor, deconv_output, exaggerate=True)

# predictions, our_features, pool_indices, sizes  = network.forward(image)


