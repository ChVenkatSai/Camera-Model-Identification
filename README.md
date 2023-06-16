# Camera-Model-Identification
Motivated by the question of whether the footage capturing a crime is genuine, we developed an algorithm as a solution to IEEE's problem on Kaggle. Our approach involves utilizing the PRNU Noise Pattern of images as input to train a Transfer Learning Architecture. 
# Motivation
Finding footage of a crime caught on tape is any investigator’s dream. But
even with crystal clear evidence, one critical question always remains–”is
the footage real?”.
# Dataset
Dataset has been taken from Kaggle Website. Train data and Test data are
explicit. We have to identify the camera model out of a list of 10 Models.
Images in the training set were captured with 10 different camera models,
a single device per model, with 275 full images from each device. The list
is as follows:
• Sony NEX-7
• Motorola Moto X
• Motorola Nexus 6
• Motorola DROID MAXX
• LG Nexus 5x
• Apple iPhone 6
• Apple iPhone 4s
• HTC One M7
• Samsung Galaxy S4
• Samsung Galaxy Note 3
Images in the test set were captured with the same 10 camera models, but
using a second device. For example, if the images in the train data for the
iPhone 6 were taken with John’s device (Camera 1), the images in the test
data were taken with John’s second device (Camera 2).
None of the images in the test data were taken with the same device as
in the train data.
You can download the datasets from : https://www.kaggle.com/competitions/sp-society-camera-model-identification/data
# Approach
CNN Architecture: Initially, you attempted to build your own CNN architecture using convolutional layers, batch normalization, pooling layers, and fully connected layers. However, the model resulted in low accuracy.

Transfer Learning: To improve the accuracy, you decided to use transfer learning. Transfer learning involves taking a pre-trained neural network model that has been trained on a different task or dataset and repurposing it for a new task. You selected five different pre-trained CNN architectures: VGG16, DenseNet201, ResNet152V2, InceptionResNetV2, and NASNetLarge. You split the training data into 80% for training and 20% for validation.

DenseNet201: Among the chosen architectures, DenseNet201 achieved the highest validation accuracy of 0.5127 after training for 5 epochs. You then selected DenseNet201 as the model to proceed with.

PRNU: Pattern Noise (PRNU) was introduced as a component for camera model identification. PRNU refers to the unique noise patterns caused by Pixel Non-Uniformity (PNU) in camera sensors. You explained that PRNU can be extracted by performing wavelet decomposition, variance estimation, denoising, and inverse wavelet transform on the noisy image.

Filters: Various filters were applied to the extracted PRNU pattern to enhance its quality and remove artifacts. These filters included a gray scale filter to combine RGB channels, a zero mean filter to make the PRNU pattern have zero mean in rows and columns, and a Wiener FFT filter to scale the frequencies based on local variances.

Training: The pre-processed PRNU patterns were then used as input data for the DenseNet201 model. The model was trained on the PRNU patterns to identify the camera model. The performance of this approach showed an improvement compared to the initial CNN architecture.
# Running the code
Note we have used kaggle paths of the competition to load the data. For example, 
train_path = '/kaggle/input/sp-society-camera-model-identification/train/train'
valid_path = '/kaggle/input/sp-society-camera-model-identification/test/test'
