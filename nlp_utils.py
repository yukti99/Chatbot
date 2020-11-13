import numpy as np
import nltk
# package for pre-trained tokenizer
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return ps.stem(word.lower())

def bagOfWords(tokenized_sentence,all_words):
    """
    sentence = ["hello","how","are","you"]
    all_words = ["hi","hello","I","you","bye","thank","cool"]
    bag_of_words =   [0,     1,     0,    1,   0,     0,      0]

    """
    tokenized_sentence = [stem(w) for  w in tokenized_sentence]
    # initialising an array of zeroes with same size as the total number of words
    bag = np.zeros(len(all_words),dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0
    return bag



"""
s = "How long does delivery take?"
print(s)
s = tokenize(s)
print(s)
words = ["organize","organizes","organizing","organized"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)
"""
"""
#checking...
sentence = ["hello","how","are","you"]
all_words = ["hi","hello","I","you","bye","thank","cool"]
bag_of_words =   bagOfWords(sentence,all_words)
print(bag_of_words)

"""