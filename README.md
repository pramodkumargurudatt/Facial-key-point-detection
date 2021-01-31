# Facial-key-point-detection
## Project introduction
This project is all about recognizing the key points in a given random facial image. The key points on a face can be eyebrows, nose lining, lips lining, etc.  
Facial key point detections are used in security systems, device authentification, employee login control, etc. 
The author has used deep learning approach to develop the software and backpropogation algorithm has been used for weight calculation.
The application is also capable of detecting the faces in a given image with bounding box. The method of Haar cascade has been used for face detection. It can handle multiple faces in a given input image.  

## Installation
Python 3,
PyTorch,
Numpy,
Matplotlib,
cv2
GPU mode (Preferred) 

## Pre-processing
**Normalization:** Converting a color image to grayscale and normalize the color range to [0,1]
**Rescale:** Rescale the image in a sample to a [224, 224] size.
**To Tensor:** Converting an image to tensor format.

## Hyperparameters
**epoch:** 60
**Learning rate:** 0.01
**Conv layers:** 2
**Batch size:** 15
**Drop out probabilities:** P1 = 0.1 and P2 = 0.2

**Optimizer:** SGD
**Loss function:** L1 loss function

## How to use the files and functions?
### Processing the images using following functions in Define the Network Architecture.ipnyb
data_transform = transforms.Compose([Rescale(250),RandomCrop(224),Normalize(),ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',root_dir='/data/training/',transform=data_transform)

### Loading the data using the following funciton in Define the Network Architecture.ipnyb
train_loader = DataLoader(transformed_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

### Training the network
def train_net(n_epochs) to be invoked for training the network.

### Visualization
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10) to be invoked for visualizing the predicted key points.

## Scope for imporvement
Transfer learning can be used to improve the model accuracy. 


