import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nlp_utils import bagOfWords, tokenize, stem
from model import NeuralNet

# opening our intents.json file
with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
pr = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # loop over all the patterns
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        # no append another array just extend it
        all_words.extend(w)
        pr.append((w,tag))

Stop_words = stopwords.words('english')
# stemming and removing unnecessary words
all_words = [stem(w) for w in all_words if w not in Stop_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
#print("T = ",tags)

# bag of words in X
X_train = []
y_train = []

for (pattern_sentence,tag) in pr:
    bag_words = bagOfWords(pattern_sentence,all_words)
    X_train.append(bag_words)
    # to get the index of the tag in tags list
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

# Creating a dataset for Training Data

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples

# Hyperparameters
# first bag of words or just all_words size will be the input_size
input_size = len(X_train[0])
print(input_size,len(all_words))
hidden_size = 8
learning_rate = 0.001
# number of passes of entire training dataset
num_epochs = 1000
output_size = len(tags)
print(output_size,tags)
batch_size =  8
dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True,num_workers = 0)

# Creating the Pytorch model and neural net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(words)
        loss = criterion(outputs,labels)
        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f'final loss,loss={loss.item():.4f}')

# saving the model
data = {
    "model_state":model.state_dict(),
    "input_size": input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags": tags
}

#  PTH is a data file for Machine Learning with PyTorch
FILE = "data.pth"
# saving to a pickled file
# to give the most flexibility later for restoring the model later.
# This is the recommended method for saving models, because it is only really necessary to save the trained model’s learned parameters.
torch.save(FILE)
print(f'Traning complete and File saved to {FILE}')








