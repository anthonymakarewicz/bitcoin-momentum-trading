import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

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
                 verbose=0,
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
        self.verbose = verbose
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

        # Training loop
        for epoch in range(self.epochs):
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
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
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
            'verbose': self.verbose,
            'patience': self.patience,
            'seq_len': self.seq_len,
            'device': self.device.type if isinstance(self.device, torch.device) else self.device
        }