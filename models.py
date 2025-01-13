import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class LSTMModel(nn.Module):
    def __init__(self, input_size, n_units=50, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, n_units, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(n_units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, timesteps, n_units)
        out = out[:, -1, :]    # Take the last timestep's output
        if self.dropout:
            out = self.dropout(out)
        out = self.fc(out)     # (batch, 1)
        return out

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 learning_rate=0.01,
                 n_units=50,
                 dropout=0.0,
                 loss='bce',        # 'bce' for binary classification
                 epochs=10,
                 batch_size=32,
                 patience=3,
                 seq_len=10,
                 n_workers=0,
                 device=None):
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.dropout = dropout
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seq_len = seq_len
        self.n_workers = n_workers

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model_ = None

    def _get_loss_function(self):
        if self.loss == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss for classification: {self.loss}")

    def _create_sequences(self, X, y=None):
        N, _ = X.shape
        if N <= self.seq_len:
            raise ValueError("Number of observations must be > seq_len.")

        X_seq = []
        if y is not None:
            y_seq = []

        for i in range(N - self.seq_len):
            X_seq.append(X[i:i+self.seq_len, :])
            if y is not None:
                y_seq.append(y[i + self.seq_len])

        X_seq = np.array(X_seq, dtype=np.float32)   # shape: (N - seq_len, seq_len, num_features)
        if y is not None:
            y_seq = np.array(y_seq, dtype=np.float32)
            return X_seq, y_seq
        return X_seq

    def build_model(self, input_size):
        model = LSTMModel(n_units=self.n_units, 
                                    input_size=input_size,
                                    dropout=self.dropout)
        return model.to(self.device)

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy().ravel()

        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)

        X_train_t = torch.tensor(X_train_seq, device=self.device)
        y_train_t = torch.tensor(y_train_seq, device=self.device).unsqueeze(-1)  # shape: (batch, 1)

        _, _, input_size = X_train_seq.shape
        if self.model_ is None:
            self.model_ = self.build_model(input_size)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers)

        optimizer = Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = self._get_loss_function()

        best_loss = np.inf
        epochs_no_improve = 0

        progress_bar = tqdm(range(self.epochs), desc="Training")
        for epoch in progress_bar:
            self.model_.train()
            epoch_loss = 0.0

            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss_val = criterion(logits, y_batch)
                loss_val.backward()
                optimizer.step()

                epoch_loss += loss_val.item() * X_batch.size(0)

            epoch_loss /= len(dataset)

            progress_bar.set_postfix({
                "Loss": f"{epoch_loss:.4f}"
            })

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

        return self

    def predict(self, X_test):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        X_test_seq = self._create_sequences(X_test)  # no y

        self.model_.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_seq, device=self.device)
            logits = self.model_(X_test_t)  # shape: (N, 1)
            # Convert logits -> probabilities -> 0/1
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()  # threshold at 0.5
            
            return preds.cpu().numpy().ravel().astype(int)

    def predict_proba(self, X_test):
        """
        Return predicted probabilities for [class 0, class 1].
        Shape: (N - seq_len, 2).
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        X_seq = self._create_sequences(X_test)
        X_seq_t = torch.tensor(X_seq, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_seq_t)       # shape: (N - seq_len, 1)
            prob1 = torch.sigmoid(logits).squeeze()  # shape: (N - seq_len,)
            prob0 = 1.0 - prob1
            # Combine into shape (N - seq_len, 2)
            probs_2d = torch.stack([prob0, prob1], dim=1)
            
        return probs_2d.cpu().numpy()

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model_ = None # rebuild the model
        return self

    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'n_units': self.n_units,
            'dropout': self.dropout,
            'loss': self.loss,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'seq_len': self.seq_len,
            'device': self.device.type if isinstance(self.device, torch.device) else self.device
        }


class CNNModel(nn.Module):
    def __init__(self, input_size, num_channels=16, kernel_size=3, dropout=0.0):
        """
        A simple 1D CNN for binary classification on time-series data.
        
        Args:
            input_size (int): Number of features per timestep.
            num_channels (int): Number of output channels for the convolution.
            kernel_size (int): Kernel size for the 1D convolution.
            dropout (float): Dropout probability.
        """
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=input_size,   # each feature is treated as a separate 'channel'
            out_channels=num_channels,
            kernel_size=kernel_size
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
        self.fc = None  # We'll define it in the forward pass on first use
        self._initialized_fc = False
        
    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        We'll conv over time dimension, so transpose -> (batch, input_size, seq_len).
        """
        # Transpose for conv1d:
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        
        out = self.conv1(x)    # (batch, num_channels, new_len)
        out = self.relu(out)
        out = self.dropout(out)
        
        batch_size, num_channels, new_len = out.shape
        out = out.reshape(batch_size, num_channels * new_len)
        
        # if fc is None, define it now:
        if self.fc is None:
            self.fc = nn.Linear(num_channels * new_len, 1)

        logits = self.fc(out)  # shape: (batch, 1)
        return logits

class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 learning_rate=0.001,
                 num_channels=16,
                 kernel_size=3,
                 dropout=0.0,
                 loss='bce',
                 epochs=10,
                 batch_size=32,
                 patience=3,
                 seq_len=10,
                 n_workers=0,
                 device=None):
        """
        A 1D CNN-based time-series binary classifier, 
        similar to LSTMClassifier but using conv layers.
        """
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seq_len = seq_len
        self.n_workers = n_workers

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model_ = None

    def _get_loss_function(self):
        if self.loss == 'bce':
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    def _create_sequences(self, X, y=None):
        """
        Takes X shape (N, num_features) 
        => returns X_seq shape (N - seq_len, seq_len, num_features).
        If y is given, produce y_seq shape (N - seq_len,).
        """
        N, _ = X.shape
        if N <= self.seq_len:
            raise ValueError("Number of observations must be > seq_len.")

        X_seq = []
        y_seq = []

        for i in range(N - self.seq_len):
            X_seq.append(X[i : i + self.seq_len, :])
            if y is not None:
                y_seq.append(y[i + self.seq_len])

        X_seq = np.array(X_seq, dtype=np.float32)  # (N - seq_len, seq_len, num_feats)
        if y is not None:
            y_seq = np.array(y_seq, dtype=np.float32)
            return X_seq, y_seq
        return X_seq

    def build_model(self, input_size):
        """
        Build a CNNModel with the given hyperparams.
        """
        model = CNNModel(
            input_size=input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        return model.to(self.device)

    def fit(self, X_train, y_train):
        """
        Fit the CNN on X_train, y_train.
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values.ravel()

        X_seq, y_seq = self._create_sequences(X_train, y_train)

        X_seq_t = torch.tensor(X_seq, device=self.device)
        y_seq_t = torch.tensor(y_seq, device=self.device).unsqueeze(-1) # (batch, 1)

        # Build model if not already
        _, _, input_size = X_seq.shape
        if self.model_ is None:
            self.model_ = self.build_model(input_size)

        dataset = TensorDataset(X_seq_t, y_seq_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers)

        criterion = self._get_loss_function()
        optimizer = Adam(self.model_.parameters(), lr=self.learning_rate)

        best_loss = np.inf
        epochs_no_improve = 0

        progress_bar = tqdm(range(self.epochs), desc="Training")
        for epoch in progress_bar:
            self.model_.train()
            epoch_loss = 0.0

            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss_val = criterion(logits, y_batch)
                loss_val.backward()
                optimizer.step()

                epoch_loss += loss_val.item() * X_batch.size(0)

            epoch_loss /= len(dataset)

            progress_bar.set_postfix({
                "Loss": f"{epoch_loss:.4f}"
            })

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

        return self

    def predict(self, X_test):
        """
        Returns 0/1 labels.
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        X_seq = self._create_sequences(X_test)
        X_seq_t = torch.tensor(X_seq, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_seq_t)  # (N - seq_len, 1)
            probs = torch.sigmoid(logits).squeeze() # (N - seq_len,)
            preds = (probs >= 0.5).float()
        return preds.cpu().numpy().astype(int)

    def predict_proba(self, X_test):
        """
        Return shape (N - seq_len, 2) => [prob(class=0), prob(class=1)].
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        X_seq = self._create_sequences(X_test)
        X_seq_t = torch.tensor(X_seq, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_seq_t)  # (N - seq_len, 1)
            prob1 = torch.sigmoid(logits).squeeze()  # shape: (N - seq_len,)
            prob0 = 1.0 - prob1
            probs_2d = torch.stack([prob0, prob1], dim=1)
        
        return probs_2d.cpu().numpy()

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model_ = None   # rebuild model next time we fit
        return self

    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'loss': self.loss,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'seq_len': self.seq_len,
            'device': self.device.type if isinstance(self.device, torch.device) else self.device
        }