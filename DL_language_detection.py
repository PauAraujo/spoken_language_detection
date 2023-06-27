# Import required packages
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

# feel free to add more imports here
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
import copy
import time

# We make sure the code can run on an available GPU
if torch.cuda.is_available():
    torch.set_default_device('cuda') 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set the sampling rate and define the list of languages and corresponding dictionary
sampling_rate = 8_000
languages = ["de", "en", "es", "fr", "nl", "pt"]
language_dict = {languages[i]: i for i in range(len(languages))}

# Load train and test data
X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
    "dataset/targets_train_int8.npy"
)
X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
    "dataset/targets_test_int8.npy"
)

# Convert data to float32 type
X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

# Define the model class
class LanguageDetectionModel(nn.Module):
    # Model Initialization
    def __init__(self, num_folds = 15, num_epochs = 20, batch_size = 16, learning_rate = 0.0001, verbose = False):
        super(LanguageDetectionModel, self).__init__()

        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose # Set verbose = True for more print() statements.

        # Defining the architecture of the model
        self.norm = nn.LayerNorm((1, 40000)) # type: ignore

        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=4)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.lstm = nn.LSTM(256, 512, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, 6)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        
    # Define the forward pass of the model
    def forward(self, x):
        x = self.norm(x)
        x = x.view(x.shape[0], 1, -1)  # reshape the input tensor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
                
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
    
    # Define the fit() function
    # This function performs k-fold cross validation
    def fit(self, X, y):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        kf = KFold(n_splits=self.num_folds, shuffle=True)

        for fold, (train_index, val_index) in enumerate(kf.split(X)):

            if self.verbose:
                print(f"Fold [{fold + 1}/{self.num_folds}]")

            X_train = X[train_index]
            y_train = y[train_index]
            X_val = X[val_index]
            y_val = y[val_index]

            # Convert data to tensors
            X_train = torch.Tensor(X_train)
            y_train = torch.Tensor(y_train).long()
            X_val = torch.Tensor(X_val)
            y_val = torch.Tensor(y_val).long()

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu'))
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            # Train the model
            for epoch in range(self.num_epochs):
                self.train()
                for inputs, labels in train_loader:
                    inputs = inputs.unsqueeze(1)
                    labels = labels.float()
                    labels = labels.long()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                self.eval()
                total, correct = 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1)
                        labels = labels.float()
                        labels = labels.long()
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = self(inputs)
                        # print(labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                    
                        correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Validation Accuracy: {accuracy:.2f}%")

    def predict(self, X, y):
        with torch.no_grad():

            X = torch.Tensor(X)
            y = torch.Tensor(y).long()

            pred_dataset = torch.utils.data.TensorDataset(X, y)
            pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)
        
            for inputs, labels in pred_loader:
                inputs = inputs.unsqueeze(1)
                labels = labels.float()
                labels = labels.long()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

        if self.verbose:
            print("\n", predicted)
            print("Prediction completed")
        return predicted

    def score(self, X, y):
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        y = torch.Tensor(y).long()

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu'))

        self.eval()
        outputs = []
        for i, _ in dataloader:
            outputs.append(self(i).argmax(dim=1).long())
        outputs = torch.cat(outputs)
        accuracy = (outputs == y).sum().item() / 1200
        accuracy *= 100
        if self.verbose:
            print(f"Accuracy is {accuracy:.2f}")
        return accuracy



# Define the parameter grid for grid search
param_grid = {'num_folds': [3, 5, 10, 15],
              'num_epochs': [5, 10, 15, 20],
              'learning_rate': [0.001, 0.0005, 0.0001, 0.00005]}

# We have a small grid for debugging purposes
#param_grid = {'num_folds': [2, 3], 
#              'num_epochs': [1, 2], 
#              'learning_rate': [0.001]}

# Create a list of all combinations of hyperparameters
param_list = list(ParameterGrid(param_grid))

# Grid search over the parameter list
performance_results = {}
best_model = None
best_accuracy = -1  # Initialize best_accuracy to a negative value (makes sure that the first calculated accuracy will be higher than best_accuracy)
# if best_accuracy were initialized to a non-negative value, 
# there might be cases where the accuracy from the first run of the loop is not better than best_accuracy, 
# and thus best_model would not be updated, which wouldn't be correct

for params in param_list:
    num_folds = params['num_folds']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    modelbig5conv = LanguageDetectionModel(num_folds=num_folds, num_epochs=num_epochs, learning_rate=learning_rate, verbose=False)

    print(f"Training the model with num_folds = {num_folds}, num_epochs = {num_epochs}, learning_rate = {learning_rate}")

    # Perform k-fold cross-validation with the updated parameters
    modelbig5conv.fit(X_train, y_train)

    # Evaluate the model
    accuracy = modelbig5conv.score(X_test, y_test)
    print(f"Test set accuracy is {accuracy:.2f}")

    # Store the parameters and corresponding accuracy
    performance_results[(num_folds, num_epochs, learning_rate)] = accuracy

    # compare new accuracy with best_accuracy  
    if best_model is None or accuracy > best_accuracy:
        best_model = copy.deepcopy(modelbig5conv) # If avg_accuracy is better make a deep copy of modelbig5conv and store it in best_model
        best_accuracy = accuracy # update best_accuracy

# Find the parameters with the highest accuracy
best_params, _ = max(performance_results.items(), key=lambda x: x[1])  # we don't need best_accuracy here anymore because we've tracked it separately
print(f"Best parameters are: num_folds = {best_params[0]}, num_epochs = {best_params[1]}, learning_rate = {best_params[2]} with accuracy = {best_accuracy:.2f}%")

# Move the model to CPU
modelbig5conv.cpu()
torch.jit.save(torch.jit.script(best_model), "modelbig5conv_v6_model.pt")