import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from torchvision import datasets, transforms

class LayerNormalization(nn.Module):
    def __init__(self,parameter_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = (int(parameter_shape[0]),int(parameter_shape[1]))
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((int(parameter_shape[0]),int(parameter_shape[1]))))
        self.beta =  nn.Parameter(torch.zeros((int(parameter_shape[0]),int(parameter_shape[1]))))

    def forward(self,input):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = input.mean(dim=dims,keepdim=True)
        var = ((input - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (input-mean)/std
        out = self.gamma * y  + self.beta
        return out
    

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values) # moze i ne mora, moze posluziti ako zatreba neka lin. transformacija
        return out

class MLP(nn.Module):

    def __init__(self, d_model,hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,patch_size_norm):
        super().__init__()
        self.normalization1 = LayerNormalization(parameter_shape=patch_size_norm)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.attention = MultiheadAttention(input_dim=d_model,d_model=d_model,num_heads=num_heads)
        self.normalization2 = LayerNormalization(parameter_shape=patch_size_norm)
        self.MLP = MLP(d_model=d_model,hidden=ffn_hidden)
    
    def forward(self,x):
        residual_x = x
        # print("Pre ulaza u Encoder, u EncoderLayeru: ",np.shape(x))
        # exit()
        x = self.normalization1.forward(x)
        #print(np.shape(x))
        x = self.attention(x, mask=None)
        x = self.dropout1(x)
        x = x + residual_x
        residual_x = x
        x = self.normalization2.forward(x)     
        x = self.MLP.forward(x)
        x = x + residual_x
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,patch_size_norm):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob,patch_size_norm=patch_size_norm )
                                     for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self,x):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return x+PE
    


class PatchingAndProjection(nn.Module):
    def __init__(self, patch_size, d_model):
        super(PatchingAndProjection, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.projection = nn.Linear(patch_size[0] * patch_size[1] * 1, d_model)  # 3 kanala za RGB slike, 1 za Grayscale

    def patching(self,folder,height,width):
        size = folder.shape[1]
        patches = []
        for img in folder:
            img_array = np.array(img) #da bih mogao da radim operacija nad tom matricom
            one_photo = []
            for i in range(0, size, height):
                for j in range(0, size, width):
                    patch = img_array[i:i+height, j:j+width]
                    flattened_patch = patch.flatten()
                    one_photo.append(flattened_patch)
            patches.append(one_photo)

        patches = np.array(patches)
        patches = torch.tensor(patches, dtype=torch.float32)
        #print("Fja. patching: ",np.shape(patches))
        return patches

    def forward(self, image):
        patches = self.patching(image, self.patch_size[0],self.patch_size[1])
        projected_patches = self.projection(patches)
        #print("Fja. forward: ",np.shape(projected_patches))
        return projected_patches


class ViT(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, number_of_classes, patch_size,batch_size):
        super().__init__()
        self.patching_and_projection = PatchingAndProjection(patch_size, d_model)
        self.positional_embeding = PositionalEncoding(d_model=d_model, max_sequence_length=int(28**2/(patch_size[0]*patch_size[1])+1))
        self.encoder = Encoder(d_model=d_model, ffn_hidden=ffn_hidden, num_heads=num_heads, drop_prob=drop_prob, num_layers=num_layers,patch_size_norm=(28**2/(patch_size[0]*patch_size[1])+1,d_model))
        self.MLP_Head = MLP(d_model=d_model,hidden=ffn_hidden)
        self.fc = nn.Linear(d_model, number_of_classes)
        self.learnable_patch = nn.Parameter(torch.randn(1, 1, d_model))
        self.batch_size = batch_size

    def forward(self, x):
        #print(np.shape(x))
        patches = self.patching_and_projection(x)
        batch_size, num_patches, _ = patches.size()
        #print("ViT,Pre: ",np.shape(patches))
        #print("Learnable parameters: ",np.shape(self.learnable_patch))
        #print("Patches Pre: ",np.shape(patches))
        # exit()

        #print(np.shape(patches)) #torch.Size([128, 49, 8])
        learnable_patch_expanded = self.learnable_patch.expand(self.batch_size, -1, -1)
        patches = torch.cat([learnable_patch_expanded, patches], dim=1)
        
        #print(np.shape(patches))
        #print(patches[2][0] == patches[1][0]) # provera za token
        #exit()

        patches = self.positional_embeding.forward(patches)

        #print("ViT,Posle: ",np.shape(patches))
        x = self.encoder.forward(patches)
        x = self.MLP_Head.forward(x)
        x = self.fc(x) # lin. transformacija do broja klasa 
        x = x[:,0,:]
        return x
        
def accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


transform = transforms.Compose([
    transforms.ToTensor(), 
])

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

filtered_data = [(img, label) for img, label in mnist_data if label in [0, 1, 2]]

images, labels = zip(*filtered_data)

images_tensor = torch.stack(images)
labels_tensor = torch.tensor(labels)


if images_tensor.dim() == 4 and images_tensor.size(1) == 1:
    images_tensor = images_tensor.squeeze(1) 
print(images_tensor.size())


num_samples = 18560

indices = torch.randperm(images_tensor.size(0))[:num_samples]

images_tensor = images_tensor[indices]
labels_tensor = labels_tensor[indices]
print(images_tensor.size())



train_dataset = TensorDataset(images_tensor, labels_tensor)


x_train = images_tensor
y_train = labels_tensor
num_classes = 3 
batch_size = 128 

transformer = ViT(d_model=8, ffn_hidden=4*8, num_heads=4, drop_prob=0.1, num_layers=4, number_of_classes=num_classes, patch_size=(4, 4), batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.0002)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10 

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0
    
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = transformer(inputs)
        
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        
        acc = accuracy(outputs, targets)
        epoch_accuracy += acc
        num_batches += 1
        
        
        #outputs1 = F.softmax(outputs, dim=-1)
        #print(outputs1[1])
        
        epoch_loss += loss.item()
    
    
    avg_accuracy = epoch_accuracy / num_batches
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")




# Evaluation of Model with Test dataset

transform = transforms.Compose([
    transforms.ToTensor(), 
])

mnist_test_data = datasets.MNIST(root='./data', train=False, download=True,transform=transform)

filtered_test_data = [(img, label) for img, label in mnist_test_data if label in [0, 1, 2]]

test_images, test_labels = zip(*filtered_test_data)

test_images_tensor = torch.stack(test_images)
test_labels_tensor = torch.tensor(test_labels)

if test_images_tensor.dim() == 4 and test_images_tensor.size(1) == 1:
    test_images_tensor = test_images_tensor.squeeze(1)

num_samples = 3072

indices = torch.randperm(test_images_tensor.size(0))[:num_samples]

test_images_tensor = test_images_tensor[indices]
test_labels_tensor = test_labels_tensor[indices]


batch_size = 128
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

transformer.eval() 
test_loss = 0
test_accuracy = 0
num_batches = 0

with torch.no_grad(): 
    for batch in test_loader:
        inputs, targets = batch
        outputs = transformer(inputs)

        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)

        test_loss += loss.item()
        test_accuracy += acc
        num_batches += 1

avg_test_loss = test_loss / num_batches
avg_test_accuracy = test_accuracy / num_batches

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
