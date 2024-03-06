import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

# Assuming input size is the size of your input features (number of cities * 2 + 1)
input_size = 30 * 2 + 1
hidden_size = 16  # Adjust based on your requirements

autoencoder = Autoencoder(input_size, hidden_size)
criterion_autoencoder = nn.MSELoss()
optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train autoencoder
num_autoencoder_epochs = 1000  # Adjust as needed

for epoch in range(num_autoencoder_epochs):
    for data in loader:  # Assuming you have a DataLoader for your dataset
        inputs, _ = data
        optimizer_autoencoder.zero_grad()
        outputs = autoencoder(inputs)
        loss_autoencoder = criterion_autoencoder(outputs, inputs)
        loss_autoencoder.backward()
        optimizer_autoencoder.step()

# Extract encoded features
encoded_features = autoencoder.encoder(x_train).detach()

# Now use the encoded features as input to your original model
class ModelWithAutoencoder(nn.Module):
    def __init__(self, input_size, h1, h2, h3, out_features):
        # ... (your existing code)

        # Additional layer to accept encoded features
        self.fc_encoded = nn.Linear(hidden_size, h1)

    def forward(self, x, encoded_features):
        x = torch.sigmoid(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)

        # Concatenate encoded features with the output of the first layer
        x = torch.cat((x, encoded_features), dim=1)

        x = torch.sigmoid(self.fc_encoded(x))
        x = self.bn_encoded(x)
        x = self.dropout_encoded(x)

        # Continue with the rest of your model
        x = torch.sigmoid(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.out(x))
        return x

model_with_autoencoder = ModelWithAutoencoder(input_size, h1, h2, h3, out_features)
criterion_with_autoencoder = nn.MSELoss()
optimizer_with_autoencoder = torch.optim.Adam(model_with_autoencoder.parameters(), lr=0.01)

# Training loop for the combined model (autoencoder + original model)
# You may need to adapt this part based on your specific requirements
# ...

# Use the combined model for prediction
# ...
