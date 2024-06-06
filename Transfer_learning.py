import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming MinDVisModel is the model from MinD-Vis and your custom fNIRS dataset is prepared
from mind_vis_model import MinDVisModel
from fnirs_dataset import FNIRSDataset

# Load pretrained MinD-Vis model
model = MinDVisModel()
pretrained_weights = torch.load('pretrained_mind_vis_weights.pth')
model.load_state_dict(pretrained_weights)

# Freeze all layers except the last few for fine-tuning
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers
for param in model.fc_layers.parameters():
    param.requires_grad = True

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Prepare DataLoader
train_dataset = FNIRSDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

print("Fine-tuning complete")

