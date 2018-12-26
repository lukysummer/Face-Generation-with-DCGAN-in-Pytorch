# Face Keypoints Detection & Applying Snapchat Filters in PyTorch, CV2

This is my implementation of a face generation algorithm, which takes in a large number of face images and generates a new face image made from scratch
from a **latent vector z (random uniform distribution in [0,1])**. Below are some examples of newly generated faces:
<img src="results.png">


There are 2 main components of this model:

1. **Discriminator**: 4-Layer CNN - Given a face image, distinguishes it as a real or a fake (generated) image

2. **Generator**: 4-Layer CNN - Given a latent vector z, generates a new face image from learned weights from images in training set. It tries to trick the 
Dircriminator to think that the generated image is REAL. 

## Repository 

This repository contains:
* **Face_Generation.py** : Complete code for implementing face generation task using DCGAN
					  
## Datasets

Datasets necessary for this implementation can be downloaded by clicking [here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip).

## List of Hyperparameters used:

* Batch Size = **128**
* Generated Image Size = **32 x 32**  
* Eength of latent vector z = **100**  
* Number of Filters in Discriminator's first hidden layer = **32**
* Number of Filters in Generator's first hidden layer = **32**
* Initial Learning Rate, [beta1, beta2] = **0.0002, [0.5, 0.999]**
* Number of Epochs = **50**

## Sources

I referenced the following sources for building & debugging the final model :

* https://github.com/udacity/P1_Facial_Keypoints/tree/master
