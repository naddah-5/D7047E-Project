import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets

from model.model import CNN
from model.deconv import DeconvCNN
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
        transforms.Resize(scale),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

image = Image.open('dataset/RSNA_DATASET_OUR_(split_into_sets)/training_set/Pneumonia/0a2f6cf6-1f45-44c8-bcf0-98a3b466b597.jpg')
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(my_device)
#print(image_tensor.shape)

predictions, our_features, pool_indices, sizes  = model.forward(image_tensor)

deconv_output = deconv_model.forward(our_features, pool_indices, input_size=image_tensor.shape[-2:], sizes=sizes)

image_tensor = image_tensor[0]
print('predicted:\n',predictions)
visualize(image_tensor, deconv_output, exaggerate=True)

# predictions, our_features, pool_indices, sizes  = network.forward(image)


