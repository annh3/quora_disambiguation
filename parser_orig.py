import csv
import os
import numpy as np
import nltk
#nltk.download('all')
from general_utils import get_minibatches
from nltk.tokenize import RegexpTokenizer

stopWords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'rights', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
stopWords_set = set(stopWords)

def parse_data(folder = os.getcwd()):
    ret = []
    folder = ""
    with open(folder + "quora_duplicate_questions.tsv", 'rb') as f:
        f.next()
        reader = csv.reader(f, delimiter = '\t')
        for row in reader:
            ret.append(row)
    return ret

def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = []
    word_to_int = {}
    model.append([0] * 50) #make this a param
    counter = 1
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]

        model.append(embedding)
        word_to_int[word]= counter
        counter = counter + 1
    print "Done.",len(model)," words loaded!"
    np_model = np.matrix(model)
    return np_model, word_to_int

def getSentenceList(sentence):
    #myWords = sentence.split()
    #myList = []
    tokenizer = RegexpTokenizer(r'\w+')
    myList = tokenizer.tokenize(sentence)

    words = [w.lower() for w in myList]
    return words

def load_and_preprocess_data(reduced, max_len=240):
    dataset = parse_data()

    total_length = len(dataset)
    lbls = [[1,0],[0,1]]
    # 2 : 1 : 1 ratio
    # create map from word to index to embedding
    embeddings, word_to_int = loadGloveModel(gloveFile='glove.6B.50d.txt') #'/Users/annhe/Documents/glove.6B/glove.6B.50d.txt'
    # create datset with paddings
    data_matrix_1 = []
    data_matrix_2 = []
    sentence_lengths_1 = []
    sentence_lengths_2 = []
    for i in range(total_length):
        if i % 10000 == 0:
            print "iteration", i
        #print i
        sentence1 = dataset[i][3]
        sentence2 = dataset[i][4]
        sentence1_vec = getSentenceList(sentence1)
        sentence2_vec = getSentenceList(sentence2)

        sentence_as_numbers_1 =[]
        sentence_as_numbers_2 =[]

        for word in sentence1_vec:
            if word in word_to_int:
                sentence_as_numbers_1.append(word_to_int[word])
            else:
                sentence_as_numbers_1.append(0)
        leftover1 = max_len - len(sentence1_vec)
        if leftover1 <= 0:
            sentence_as_numbers_1 = sentence_as_numbers_1[:max_len]
            leftover1 = 0
        #pad leftover with 0
        for _ in range(leftover1):
            sentence_as_numbers_1.append(0)
        # append the length of sentence 1
        sentence_lengths_1.append(min(len(sentence1_vec), max_len))

        for word in sentence2_vec:
            if word in word_to_int:
                sentence_as_numbers_2.append(word_to_int[word])
            else:
                sentence_as_numbers_2.append(0)
        leftover2 = max_len - len(sentence2_vec)
        if leftover2 <= 0:
            sentence_as_numbers_2 = sentence_as_numbers_2[:max_len]
            leftover2 = 0
        #pad leftover with 0
        for _ in range(leftover2):
            sentence_as_numbers_2.append(0)
        # append the length of sentence 2
        sentence_lengths_2.append(min(len(sentence2_vec), max_len))


        data_matrix_1.append((sentence_as_numbers_1, lbls[int(dataset[i][5])]))
        data_matrix_2.append((sentence_as_numbers_2, lbls[int(dataset[i][5])]))
        # if i % 10000 == 0:
        #     print sentence1
        #     print sentence1_vec
        #     print sentence2
        #     print sentence2_vec
        #     print((sentence_as_numbers, int(dataset[i][5])))

    train_set_end = int(total_length * 0.7)
    train1 = data_matrix_1[:train_set_end]
    dev_set_end = train_set_end + int(total_length * 0.15)
    dev1 = data_matrix_1[train_set_end :dev_set_end]
    test1 = data_matrix_1[dev_set_end:]
    train2 = data_matrix_2[:train_set_end]
    dev2 = data_matrix_2[train_set_end :dev_set_end]
    test2 = data_matrix_2[dev_set_end:]
    train_lengths = (sentence_lengths_1[:train_set_end], sentence_lengths_2[:train_set_end])
    dev_lengths = (sentence_lengths_1[train_set_end:dev_set_end], sentence_lengths_2[train_set_end:dev_set_end])
    test_lengths = (sentence_lengths_1[dev_set_end:], sentence_lengths_2[dev_set_end:])
    if reduced:
        train1 = train1[:20000]  # was 20000 #was 10000
        dev1 = dev1[:5000] # was 5000
        test1 = test1[:5000] # was 5000
        train2 = train2[:20000] # was 20000 #was 10000
        dev2 = dev2[:5000]
        test2 = test2[:5000]

    return embeddings, train1, dev1, test1, train2, dev2, test2, train_lengths, dev_lengths, test_lengths

def minibatches(data1, data2, data_sizes, batch_size):

    one_hot = []
    x = []
    for i in range(len(data1)):
        lbl = data1[i][1]
        one_hot.append(data1[i][1])
        x.append((data1[i][0], data2[i][0], data_sizes[0][i], data_sizes[1][i]))

    return get_minibatches([x,one_hot], batch_size)
