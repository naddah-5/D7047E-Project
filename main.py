#from dataset.dataset import Dataset
from model.model import CNN
from application.train import Training
from application.test import test_model
import torch

from dataset.dataset import load_dataset

from model.deconv import DeconvCNN
import matplotlib.pyplot as plt


def main():
    epochs = 1
    batch_size = 10
    learning_rate = 0.0001
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)
    
    train_loader, validation_loader, test_loader= load_dataset()
    

    model = CNN(class_count=2, device=my_device)


    training = Training(model, train_loader, validation_loader, test_loader, epochs, learning_rate, device=my_device)
    training.train_model()
    
    ########### Deconv starts here

    # Initialize the DeconvCNN with the trained CNN's weights
    deconv_model = DeconvCNN(class_count=2, device=my_device)
    deconv_model.classifier.load_state_dict(model.classifier.state_dict())
    deconv_model.features[0].weight.data = model.features[4].weight.data
    deconv_model.features[2].weight.data = model.features[2].weight.data
    deconv_model.features[4].weight.data = model.features[0].weight.data

    # Choose an input image from the test_loader
    input_image, _ = next(iter(test_loader))
    input_image = input_image.to(my_device)

    # Pass the input image through the CNN to obtain feature maps and pooling indices
    with torch.no_grad():
        _, indices1, indices2 = model(input_image)

    # Pass the feature maps and pooling indices through the DeconvCNN
    with torch.no_grad():
        reconstructed_images = deconv_model(_, indices1, indices2, input_image.size())

    # Visualize the input image and reconstructed images


    def display_image(image_tensor, title):
        image = image_tensor.squeeze().cpu().numpy()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
    
    def display_image(image_tensor, title):
        image = image_tensor.squeeze().cpu().numpy()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    # Display the input image
    display_image(input_image[0], "Input Image")

    # Display the reconstructed images
    for i in range(reconstructed_images.size(0)):
        display_image(reconstructed_images[i], f"Reconstructed Image {i+1}")


    #test_accuracy = test_model(test_loader=test_loader, network=model, device=my_device)

    #print("\nTest accuracy: %f" % test_accuracy)

if __name__=="__main__":
    main()
