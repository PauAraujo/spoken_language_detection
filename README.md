# Spoken language detection
The objective of this project is to employ deep learning techniques to categorize the language of an audio clip. The dataset comprises one thousand audio clips for each of the following languages: German, English, Spanish, French, Dutch, and Portuguese.

## Data loading and processing
The data was loaded from Numpy arrays and converted to PyTorch tensors for training the deep learning model. The nn.LayerNorm was chosen for normalization as it standardizes the features across the temporal dimension of each individual sample, instead of normalizing across the entire batch. This ensures stable training and generalization regardless of the batch size, and it’s particularly beneficial in the presence of varying speech patterns and intensities among different languages.

## Architecture design
The model combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) units. It consists of five convolutional layers, with a kernel size of 9, stride of 1, and padding of 4 (Figure 1). Five layers give the best balance between model size and accuracy. Each layer is followed by batch normalization, ReLU activation and max pooling. The number of filters is progressively doubled. This structure helps extract relevant features and reduce dimensionality. The output is reshaped and passed to a bidirectional LSTM layer with 512 hidden units. The bidirectionallity allows the LSTM to use information from both directions in the model. In addition, a dropout layer of 0.2 was added to prevent overfitting. The final fully connected layer maps to six output units, representing the languages, with a softmax function providing class probabilities. Cross Entropy Loss provides a way to quantify the difference between predicted probabilities and the true class labels in a multiclass classification problem.

<img src="https://github.com/PauAraujo/spoken_language_detection/blob/main/other/NN_architecture.jpg?raw=true" width="500">
Figure 1: Architecture for multilingual language detection

## Experiment 
K-fold cross-validation (between 3 and 15) was employed in the experiments to provide robustness against overfitting during the training and validation of the model. Hyperparameters, including learning rates (ranging from 0.00005 to 0.001) and the number of epochs (between 5 and 20), were tuned to optimize model performance. Moreover, different model architectures, with varying numbers of convolutional layers (from 1 to 5), were compared to discern the most effective structure.

## Results
The model achieved an overall accuracy of 86.5% on the test data and 92% on the competition server. The best configuration involved 20 epochs, 15 folds, and a learning rate of 0.0001. Precision, recall, F1 score, and accuracy for each language class are presented in Table 1, while Table 2 displays the confusion matrix. Performance varied across languages, with French having the highest accuracy of 91% and English the lowest at 72%. Spanish exhibited the highest precision (94%), while Dutch had the lowest (72%). Remarkably, French showed the highest recall rate at 91%, while English had the lowest at 72%. F1 scores ranged from 91% (Spanish) to 75% (Dutch). The model excelled at identifying Spanish and French but faced challenges with Dutch and English. These findings suggest potential areas for improvement, such as additional feature extraction or improved handling of languages with complex phonetics or similar linguistic characteristics.

Table 1: Performance scores for each language class.

| Language | Precision | Recall | F1 Score | Accuracy |
|----------|-----------|--------|----------|----------|
| de       | 0.82      | 0.84   | 0.83     | 0.84     |
| en       | 0.82      | 0.72   | 0.76     | 0.72     |
| es       | 0.94      | 0.90   | 0.91     | 0.90     |
| fr       | 0.85      | 0.91   | 0.88     | 0.91     |
| nl       | 0.72      | 0.77   | 0.75     | 0.77     |
| pt       | 0.85      | 0.85   | 0.85     | 0.85     |


Table 2: Confusion matrix for the six predicted classes.
|     | de  | en  | es  | fr  | nl  | pt  |
|-----|-----|-----|-----|-----|-----|-----|
| de  | 168 | 9   | 0   | 4   | 17  | 2   |
| en  | 12  | 144 | 2   | 6   | 24  | 12  |
| es  | 3   | 3   | 179 | 6   | 3   | 6   |
| fr  | 5   | 4   | 3   | 182 | 3   | 3   |
| nl  | 12  | 13  | 1   | 12  | 154 | 8   |
| pt  | 5   | 2   | 6   | 5   | 12  | 170 |

## Conclusions
- The proposed architecture effectively captures both the spectral and temporal patterns in the audio data. CNN layers extract local features like phonetic anomalies while the LSTM layer captures long-range dependencies like context based features of a sentence, illustrating the strength of such a hybrid model for audio classification tasks.
- The model's difficulty distinguishing Dutch, English, and German could stem from shared phonetic and prosodic features among these languages or how they are represented in the dataset.
- Techniques such as data augmentation or additional feature engineering could be employed to improve performance on these classes. 




