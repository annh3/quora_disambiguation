import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools
from parser import parse_data
import random
import pickle

random.seed(42)

stopWords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'rights', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
stopWords_set = set(stopWords)

def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]

        model[word] = np.asarray(embedding)
    print "Done.",len(model)," words loaded!"
    return model

def getFeatureVector(sentence1, sentence2, wordVectors, embedding_length):
    sentVector1 = np.zeros((embedding_length,))
    count_len = 0
    for word in sentence1:
        if word in wordVectors:
            sentVector1 += wordVectors[word]
            count_len += 1
    if count_len > 0:
        sentVector1 = sentVector1 * (1/float(count_len))
    sentVector2 = np.zeros((embedding_length,))
    count_len = 0
    for word in sentence2:
        if word in wordVectors:
            sentVector2 += wordVectors[word]
            count_len += 1
    if count_len > 0:
        sentVector2 = sentVector2 * (1/float(count_len))
    return np.concatenate((sentVector1, sentVector2), axis=0), np.concatenate((sentVector2, sentVector1), axis=0)

def getSentenceList(sentence):
    myWords = sentence.split()
    myList = []
    for word in myWords:
        word = word.lower()
        word = "".join(c for c in word if c.isalnum())
        if word not in stopWords_set:
            myList.append(word)
    return myList


def getSentenceFeatures(indices, dataset, wordVectors, embedding_length):
    x_features_list = [] #make these numpy
    y_labels_list = []
    for index in indices:
        sentence1 = dataset[index][3]
        sentence2 = dataset[index][4]
        sentencelist1 = getSentenceList(sentence1)
        sentencelist2 = getSentenceList(sentence2)
        featureVector1, featureVector2 = getFeatureVector(sentencelist1, sentencelist2, wordVectors, embedding_length)
        x_features_list.append(featureVector1)
        x_features_list.append(featureVector2)
        y_labels_list.append(dataset[index][5])
        y_labels_list.append(dataset[index][5])
    return x_features_list, y_labels_list

def getTrainTestSets(gloveFile='/Users/annhe/Documents/glove.6B/glove.6B.50d.txt'):
    dataset = parse_data()
    dataset.pop(0)
    #train_set, dev_set, test_set = split_dataset()
    total_length = len(dataset)
    test_size = 101072
    #print dataset[0][0]
    #print total_length * (0.25)
    totalIndices = set(range(0,total_length))
    testIndices = random.sample(range(0,total_length),test_size)
    remaining = totalIndices - set(testIndices)
    trainIndices = list(remaining)

    pickle.dump(testIndices, open('testIndices.txt', 'wb'))

    wordVectors = loadGloveModel(gloveFile)
    #print type(wordVectors['the'])
    #print wordVectors['the']
    embedding_length = 50
    X_train, y_train = getSentenceFeatures(trainIndices, dataset, wordVectors, embedding_length)
    X_test, y_test = getSentenceFeatures(testIndices, dataset, wordVectors, embedding_length)
    return X_train, y_train, X_test, y_test
