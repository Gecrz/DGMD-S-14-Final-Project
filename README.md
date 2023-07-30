# DGMD-S-14-Final-Project

## Computer Vision Modeling for Identification of Benign vs Malignant breast cancer using Histopathology Imagery


### In a nutshell

Breast cancer is a significant global health concern, necessitating the development of accurate and efficient diagnostic tools. In this project, we propose a novel image classification modeling approach for the identification of Benign vs Malignant breast cancer using histopathology imagery. The purpose of this project is to utilize advanced computer vision techniques to classify and analyze histopathology images. 

Overall, our CNN architecture is designed to learn discriminative features from medical images of tumors and make accurate predictions regarding tumor malignancy. It employs convolutional layers for feature extraction, dropout for regularization, and sigmoid activation for binary classification. By combining these techniques, the model aims to achieve robust performance and generalization on unseen data, enabling accurate identification of malignant and benign tumors from medical images.

### How to use the code
Final notebook: https://github.com/Gecrz/DGMD-S-14-Final-Project/blob/main/Final%20Model/Computer_Vision_Model_for_Breast_Cancer.ipynb

There are several steps in the modeling process: Data ingestion, data processing, model training, validation and predictions. 

##### Import Required Libraries

The necessary libraries for building and training the model need to be imported, including TensorFlow, Keras, NumPy, Pandas, Matplotlib, scikit-learn, requests, and tqdm.

##### Data Ingestion

Download the dataset (BreaKHis_v1.tar.gz) using the requests library. The dataset contains histopathological images of breast tissue. The downloaded dataset is then extracted to obtain 400x PNG image files. Link to the dataset: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

##### Data Preparation and Visualization

The code creates a DataFrame that contains the filenames of the images, along with the corresponding labels (malignant or benign) and patient IDs. The DataFrame is then split into training, validation, and test sets.
Data Preprocessing and Augmentation: The code uses ImageDataGenerator from Keras to perform data preprocessing and augmentation. It rescales the pixel values of the images to the range [0, 1], applies random rotations (between -10 and +10 degrees), and performs horizontal and vertical shifts to augment the data.

##### Model Architecture

We were able to optimize our baseline model by defining and designing a convolutional neural network (CNN) model using the Sequential API from Keras. The model consists of several convolutional layers, batch normalization layers, dropout layers, and dense layers. The final layer uses the sigmoid activation function to produce the binary classification output.

##### Model Training

We compiled the model using the Adam optimizer and binary cross-entropy loss function. It is then trained on the training data using the fit method. Early stopping is implemented to stop training if the validation loss does not improve for a specified number of epochs.

##### Model Evaluation

The model's performance is evaluated on the validation set, and the training and validation loss and accuracy are recorded during the training process. Predictions were made using the test data set. We then generated a classification report, confusion matrix and ROC AUC Score. 

### Results

The model was trained for 15 epochs using a binary cross-entropy loss function and the Adam optimizer. During training, the training accuracy increased from approximately 73.82% in the first epoch to around 88.19% in the final epoch. The validation accuracy, on the other hand, showed fluctuating behavior, starting at 60% and reaching 96.52% in the last epoch, due to an initial low learning rate. The validation loss decreased over the epochs, indicating that the model improved its generalization capabilities.

Classification Report:


              precision    recall  f1-score   support

           0       0.96      0.69      0.80       103
           1       0.79      0.97      0.87       120

    accuracy                           0.84       223

    macro avg      0.87      0.83      0.84       223

    weighted avg   0.87      0.84      0.84       223

After training, the model was evaluated on the test set, which contained 223 samples. The classification report above shows that the model achieved an overall accuracy of 84% on the test set. The precision for identifying malignant tumors was 0.79, while the recall (sensitivity) was 0.97, indicating that the model is good at correctly identifying malignant cases. For benign tumors, the precision was 0.96, and the recall was 0.69, showing that the model has higher accuracy in identifying benign cases but may classify some negative cases as positive. 

The model's performance was further evaluated using the ROC-AUC score, which measures the model's ability to distinguish between the two classes. The model achieved an ROC-AUC score of 0.9312, indicating good discrimination power between malignant and benign cases.

### Conclusion

In conclusion, the trained model achieved a reasonable level of accuracy in classifying malignant and benign tumors, with a satisfactory ROC-AUC score. However, there is room for improvement, especially in improving the recall for benign cases. The model did great identifying positive cases, which is ideal in our case, as missing malignant tumors can have significant consequences in medical applications. Further fine-tuning, addressing class imbalances or exploring more complex model architectures may lead to improved results.

### Dependencies

The model was trained using Google Colab and Python Programming Language. 


