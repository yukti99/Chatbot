import random
import torch
import json
from model import NeuralNet
from nlp_utils import bagOfWords, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')4
with open('intents.json','r') as f:
    intents = json.load(f)


FILE = "data.pth"
data = torch.load(FILE)
model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
# call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
# acts like a switch to turn of some layers during evaluation/inference
# this is evaluation mode
model.eval()

bot_name = "Yukti's Bot"
print("Let's chat! Type 'quit' to exit..")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    # bagofwords function returns a numpy array
    X = bagOfWords(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _,predicted = torch.max(output,dim=1)
    # predicted.item - class label
    tag = tags[predicted.item()]

    # checking if the probability of the tag is high enough
    # applying softmax to get the actual probabilities
    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]
    if (prob.item() > 0.70):
        # finding corresponding intent for this tag
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name} : {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name} : Sorry! I do not understand...")






