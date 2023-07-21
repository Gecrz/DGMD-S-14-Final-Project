# DGMD-S-14-Final-Project

## Project Title: Image Classification Modeling for Identification of breast cancer using Histopathology Imagery

## Group Members: Yaseen Khan Mohmand, Amreen Khureshi, Dongshen Peng, George Cruz

### Type of Project: Engineering 

Summary: Breast cancer is a significant global health concern, necessitating the development of accurate and efficient diagnostic tools. In this project, we propose a novel image classification modeling approach for the identification of breast cancer using histopathology imagery. The purpose of this project is to utilize deep learning techniques to classify and analyze histopathology images, thereby aiding in the automated early cancer detection and precision diagnosis. 

### Level: Graduate

#### Introduction: 

Breast cancer is the most common type of cancer among women worldwide. Upwards of 2 million women were diagnosed with breast cancer in 2020, in the same year 684,996 deaths were recorded due to breast cancer. When detected and treated early, breast cancer has a high survival rate. There are multiple methods for detecting breast cancer, such as mammograms, Ultrasounds, MRIs and Biopsies. For the purposes of our project we will be focusing on improving detection of breast cancer using deep learning techniques for viewing microscopic images of biopsies.
In the Biopsy procedure a small sample of breast tissue is taken and then examined by a trained pathologist under a microscope. It provides vital information about the nature of suspicious lumps or abnormalities discovered through other screening methods. Biopsies provide information on stage of the cancer, whether it’s benign or malignant and once the treatment is provided whether there is disease still left. 
The Biopsy procedure is however, expensive, complex, and time consuming. This is because it requires trained Pathologists, to look at the biopsy samples under a microscope. This is a complex and time consuming procedure as each part of the sample needs to be analyzed carefully to avoid the chances of a false negatives.
Our project’s goal is to show that deep learning techniques can augment the capabilities of pathologists in the analysis of biopsy images, by doing so, we not only seek to minimize the risk of potential inaccuracies but also aim to reduce the overall cost associated with biopsy procedures. Ultimately, our endeavor could pave the way for quicker, more accurate biopsy results, thereby expanding the accessibility of these vital procedures to a larger patient population. We will be using publicly available datasets for this project.

#### Background: 

	Histopathology is the study of diseased cells and tissues using a microscope, and the purpose of this project is to process these microscoped images to automate disease detection. The first model we will be using is a baseline Convolutional Neural Network. CNNs are typically used in image processing because of their ability to capture spatial dependencies in data. This will give us a baseline of model accuracy to move forward with. We will also use transfer learning models; these models are typically pre-trained on large datasets, which allows us to move forward without much hyperparameter tuning. We aim to use 2-3 pre-trained models, and compare them to our baseline CNN.
	Another model that we may consider is a Siamese model. A siamese model would be beneficial if there is a class imbalance within the data, or if the dataset is smaller than anticipated (this is to be determined after data import). A Siamese model works by comparing pairs of images to determine their similarity. This will be useful to identify micro-differences between diseased and non-diseased cells. 
	Because we are working with medical data, it’s important to not only identify true positives, but also false positives and false negatives. For this reason, we will use an F1 score and a confusion matrix to evaluate each of our models. Our aim here is to maximize our F1 score, and to minimize false positive/negative counts.

#### Literature Review:
Artificial Intelligence–Based Breast Cancer Nodal Metastasis Detection: This work was done in 2019. The authors of the article evaluated the performance of a deep learning-based AI algorithm called LYmph Node Assistant or LYNA for the detection of metastatic breast cancer in sentinel lymph node biopsies. They were able to achieve a “higher tumor-level sensitivity than, and comparable slide-level performance to, pathologists”. Some interesting things to note were, they used whole slide images rather than multiple zoomed in images of the biopsy. They also used multiple images from the same patient’s biopsy.

The authors of the article highlighted several key findings from the studies they reviewed. First, AI algorithms can achieve high levels of sensitivity and specificity for the detection of metastatic cancer. Second, AI algorithms are not affected by common histology artifacts, such as overfixation, poor staining, and air bubbles. Third, AI algorithms can be used to exhaustively evaluate every tissue patch on a slide, which can help to reduce the number of false negatives.
You Only Look Once: Unified, Real-Time Object Detection: You Only Look Once (‘YOLO’) is a unified, real-time object detection system that unifies the detection pipeline into a single neural network. In this neural network, the initial convolutional layers extract features from the image while the fully connected layers predict the output probabilities. This is different from classic object detection because YOLO reasons globally about an image, allowing it to implicitly encode contextual information about classes. We will look into the applications of YOLO on histopathology imagery.
Breast Cancer Histopathology Image Classification Using an Ensemble of Deep Learning Models: This paper was published online 2020 Aug 5. This paper presents an ensemble deep learning approach for the definite classification of non-carcinoma and carcinoma breast cancer histopathology images using the collected dataset. For the individual and ensemble models, they selected 80% of images for training and the remaining 20% for testing purposes with the same percentage of carcinoma and non-carcinoma images.

They trained four different models based on pre-trained VGG16 and VGG19 architectures and followed an ensemble strategy by taking the average of predicted probabilities. As for the results, the ensemble of fine-tuned VGG16 and VGG19 offered sensitivity of 97.73% for carcinoma class, overall accuracy of 95.29% and an F1 score of 95.29%. 

The limitations of this study: their collected dataset is comparatively small in contrast to the datasets used in numerous state-of-the-art studies. Also, their dataset contains merely two-class images. 
Deep Learning to Improve Breast Cancer Detection on Screening Mammography:This research developed a deep learning algorithm that accurately detects breast cancer on screening mammograms. The algorithm utilizes a training approach that requires initial lesion annotations but subsequently only relies on image-level labels, eliminating the need for rare lesion annotations. According to the research paper, the algorithm achieved excellent performance on independent test sets, demonstrating its potential to improve clinical tools for reducing false positive and false negative results in breast cancer screening.
Breast cancer detection using deep learning: Datasets, methods, and challenges ahead: The paper reviews previous studies that have utilized machine learning, deep learning, and deep reinforcement learning techniques for breast cancer classification and detection. It also discusses the availability of publicly accessible datasets for different imaging modalities, which can facilitate future research in the field. The authors emphasize the need for external validations of AI models to ensure their reliability and efficacy as clinical decision-making tools. The paper concludes with a critical discussion on the challenges and prospects for future research in breast cancer detection using deep learning, highlighting the limitations of current approaches.


#### Methodology:

Sourcing data from Kaggle (July 5)
Cleaning and importing the data (July 10)
Determine how we want to prep the data (do we want to group the images?)
Determine if we want to use all the images, or a smaller subset
Create train and test sets (July 13)
Create a baseline CNN classifier (July 17)
Try various loss functions, activation functions, and epochs
Use transfer learning methods to generate 2-3 models (July 21)
Possible models include: MobileNetV1, ResNet50
Consider an additional Siamese model (July 24)
Do this if there is a class imbalance or small number of images overall
Generate evaluation metrics (July 28)
Possible metrics include F1 score and confusion matrix

#### Division of Labor:

Yaseen:
Importing/cleaning data:
 grouping the data
 creating a subset of the data
Creating one transfer learning model
Completing project check in
Amreen: 
Sourcing data
Creating the initial baseline CNN classifier:
tuning hyperparameters (with Dongshen)
Building a Siamese model, if we decide to do this based on class balancing (with George)
Finalizing report and deliverable
Dongshen:
Creating train and test sets
Helping tune hyperparameters for baseline CNN (with Amreen)
Creating one transfer learning model
Organizing code for final deliverable
George:
Creating one transfer learning model
Helping build Siamese model (with Amreen)
Generating evaluation metrics to compare all models
Organizing final report
