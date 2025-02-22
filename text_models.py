import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from opacus.layers import DPLSTM

class ImdbFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ImdbFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()  # For binary output

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class TextFNN(nn.Module):
    def __init__(self, input_dim=300, num_classes=5):
        super(TextFNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(self.dropout(x))))
        x = self.fc3(x)
        return x

class LargeTextCNN(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 embedding_dim=300,
                 max_length=100,
                 num_filters=100,
                 dropout_rate=0.5,
                 num_classes=1,
                 embedding_matrix=None):
        """
        Initializes the TextCNN model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            max_length (int): Maximum length of input sequences.
            num_filters (int): Number of filters per convolution.
            dropout_rate (float): Dropout rate for regularization.
            num_classes (int): Number of output classes. Use 1 for binary classification.
            embedding_matrix (np.array, optional): Pre-trained embedding weights.
        """
        super(LargeTextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Embedding layer: initialize with or without pre-trained weights.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            # embedding_matrix should be a NumPy array of shape (vocab_size, embedding_dim)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # Set to True if you wish to fine-tune.

        # Define convolution layers with multiple kernel sizes.
        kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected output layer.
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_length) containing word indices.

        Returns:
            torch.Tensor: Output tensor. For binary classification, the output is passed
                          through a sigmoid activation.
        """
        # x: (batch_size, max_length) --> (batch_size, max_length, embedding_dim)
        x = self.embedding(x)

        # Permute to (batch_size, embedding_dim, max_length) for Conv1d.
        x = x.permute(0, 2, 1)

        # Apply convolution + ReLU + global max pooling for each kernel size.
        conv_results = []
        for conv in self.convs:
            conv_x = F.relu(conv(x))  # Shape: (batch_size, num_filters, L_out)
            # Global max pooling over the time dimension.
            pooled = F.max_pool1d(conv_x, kernel_size=conv_x.shape[2]).squeeze(2)  # (batch_size, num_filters)
            conv_results.append(pooled)

        # Concatenate pooled features: shape becomes (batch_size, num_filters * len(kernel_sizes))
        cat = torch.cat(conv_results, dim=1)
        cat = self.dropout(cat)

        # Fully connected layer.
        logits = self.fc(cat)

        # For binary classification, apply sigmoid activation.
        if self.num_classes == 1:
            output = torch.sigmoid(logits)
        else:
            output = logits  # For multi-class, output raw logits (softmax can be applied externally if needed).

        return output


class TextCNN(nn.Module):
    def __init__(self, embedding_dim=300, num_classes=5):
        super(TextCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, 256)

        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=4, padding=2)
        self.gn2 = nn.GroupNorm(32, 256)

        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=5, padding=2)
        self.gn3 = nn.GroupNorm(32, 256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3, 512)
        self.gn_fc = nn.GroupNorm(32, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x1 = self.gn1(F.relu(self.conv1(x))).max(dim=2)[0]
        x2 = self.gn2(F.relu(self.conv2(x))).max(dim=2)[0]
        x3 = self.gn3(F.relu(self.conv3(x))).max(dim=2)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(F.relu(self.gn_fc(self.fc1(x))))
        return self.fc2(x)

class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_classes=5):
        super(BiLSTMAttention, self).__init__()
        self.lstm = DPLSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(2), dim=1)
        attn_output = torch.sum(lstm_out * attn_weights.unsqueeze(2), dim=1)
        x = self.dropout(F.relu(self.fc1(attn_output)))
        return self.fc2(x)


class TinyTransformer(nn.Module):
    def __init__(self, embedding_dim=300, num_heads=4, num_classes=5):
        super(TinyTransformer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=512)
        self.transformer = TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)