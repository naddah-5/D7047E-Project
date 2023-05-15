import cv2
import numpy as np


def visualize(original_tensor, deconv_output, exaggerate: bool = True):
    
    # original_image = original_tensor.squeeze().cpu().numpy()
    # if original_image.ndim == 2:  # If it's a 2D array (H, W)
    #     original_image = np.expand_dims(original_image, axis=2)  # Convert it to a 3D array (H, W, C)
    # original_image = np.transpose(original_image, (1, 2, 0))  # Now it can be transposed


    deconv_image = deconv_output[0].cpu().detach().numpy()
    deconv_image = np.transpose(deconv_image, (1, 2, 0))  # only if your tensor is in (C, H, W) format
    #print(deconv_image.shape, np.max(deconv_image))
    
    if exaggerate == True:
        deconv_image = (deconv_image * 255/(np.max(deconv_image))).astype(np.uint8)  # only the tensor is normalized
    else:
        deconv_image = (deconv_image * 255).astype(np.uint8)

    #print(deconv_image.shape, np.max(deconv_image))
    




    
    original_image = original_tensor.cpu().detach().numpy()
    original_image = np.transpose(original_image, (1, 2, 0))  # only if your tensor is in (C, H, W) format
    #print(original_image.shape)#, np.max(original_image))

    if exaggerate == True:
        original_image = (original_image * 255/(np.max(original_image))).astype(np.uint8)  # only the tensor is normalized
    else:
        original_image = (original_image * 255).astype(np.uint8)
    #print(original_image.shape, np.max(original_image))
    
    
    combined_images=np.hstack((original_image ,deconv_image))

    #cv2.imshow('deconv', deconv_image)
    #cv2.imshow('original',original_image)
    cv2.imshow('Original/Deconv',combined_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()