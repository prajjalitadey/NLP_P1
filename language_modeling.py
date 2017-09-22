from collections import defaultdict
from pprint import pprint
import pandas as pd
from random import randint, randrange
import time
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


# create bigram table
def store_counts(filename):
    text_file = open(filename, 'r')
    lines = text_file.readlines()
    types = defaultdict(lambda: defaultdict(int))
    seen = set()  # instantiate seen word set

    for line in lines:
        tokens = line.replace(" n't", "n 't")  # Standardize the contractions ('t is a separate word)
        tokens = line.replace('-', ' ')  # Get rid of hyphens
        tokens = ['<s>'] + line.split() + ['</s>']

        count = len(tokens)
        for i in range(count-1):
            # treat upper and lower case words the same
            word1 = tokens[i].lower()
            word2 = tokens[i+1].lower()

            #if words NOT in set, add to set, and change current word to <unk>
            if(not(word1 in seen)):
                seen.add(word1)
                word1 = "<unk>"
            
            if(not(word2 in seen)):
                seen.add(word2)
                word2 = "<unk>"

            types[word1][word2] += 1

    # convert dictionary to table
    table = pd.DataFrame(types).T
    # add totals
    table['SUM'] = table.sum(axis=1)
    table.loc['</s>', 'SUM'] = int(table.loc['<s>', 'SUM'])
    table = table.fillna(0).applymap(lambda x: int(x))
    return table


# return the unigram P(word) for a given word, with a given table of counts
def unigram(word, table):
    try:
        return float(table.loc[word, 'SUM'])/float(table['SUM'].sum())
    except KeyError:
        print "This word doesn't exist in the corpus."


# return Laplace smoothed unigram probability
def smoothedUnigram(word, table):
    try:
        if word not in list(table.columns.values):
            word = '<unk>'
        return float(table.loc[word, 'SUM']+1)/float(table['SUM'].sum()+len(table.columns))
    except KeyError:
        print "This word doesn't exist in the corpus."

def KneserNeyUnigram(word, table):
    try:
        if word not in list(table.columns.values):
            word = "<unk>"
        return 
    except KeyError:
        print "This word does not exist in the corpus"


# return the bigram P(word2|word1) for the given table of counts
def bigram(word1, word2, table):
    try:
        if word not in list(table.columns.values):
            word = '<unk>'
        return float(table.loc[word1, word2])/float(unigram(word1, table))
    except KeyError:
        print "Either word1 or word2 doesn't exist in the corpus"


# return Laplace smoothed bigram probability
def smoothedBigram(word1, word2, table):
    try:
        if word1 not in list(table.columns.values):
            word1 = '<unk>'
        if word2 not in list(table.columns.values):
            word2 = '<unk>'
        return float(table.loc[word1, word2]+1)/float(unigram(word1, table)+len(table.columns))
    except KeyError:
        print "Either word1 or word2 doesn't exist in the corpus."


# create sum list
def sumList(table):
    sums = table["SUM"]
    tokens = sums.index.tolist()

    wordprob = []

    for i in range(0, len(sums)):
        currentcount = sums[i]
        currentword = tokens[i]
        for i in range(0, int(currentcount)):
            wordprob.append(currentword)

    return wordprob

# create random unigram sentence
def rsgUnigram(table):
    endgram = False
    wordlist = sumList(table)
    # return wordlist
    sentence = wordlist[randrange(0, len(wordlist)-1)]
    while(not endgram):
        currentword = '<s>'
        while (currentword == '<s>'):
            currentindex = randrange(0, len(wordlist)-1)
            currentword = wordlist[currentindex]
        sentence = sentence + " " + currentword
        if currentword == '</s>':
            endgram = True

    return sentence.split('</s>')[0].strip()


def bigram_sentence_generator(counts):
    counts = counts.drop('SUM', axis=1)
    sentence = ''
    token = '<s>'
    while (token != '</s>'):
        row = counts.loc[token]  # get counts based on previous token
        token_list = []  # create a list of tokens based on counts
        for label, value in row.iteritems():
            token_list.extend([label] * int(value))
        token = '<s>'
        while(token == '<s>'):
            rd_idx = randint(0, len(token_list) - 1)  # pick random index
            token = token_list[rd_idx]  # get corresponding token
        sentence += ' ' + token
    return sentence.split('</s>')[0].strip()


# returns perplexity using unigram model
def uniPerplexity(trainTable, testTable):
    wordProb = 0
    testWords = sumList(testTable)

    for i in range(0, len(testWords)):
        currentword = testWords[i]
        wordProb = wordProb - math.log(smoothedUnigram(currentword, trainTable))

    return (math.exp(wordProb/len(testWords)))


# returns perplexity using bigram model
def biPerplexity(trainTable, filename=None, line=None):
    if filename:
        text_file = open(filename, 'r')
        lines = text_file.readlines()
    if line:
        lines = line
    total_tokens = 0
    wordProb = 0

    for line in lines:
        #Standardize the contractions ('t is a separate word)
        tokens = line.replace(" n't", "n 't")
        #Get rid of hyphens
        tokens = line.replace('-', ' ')
        tokens = ['<s>'] + line.split() + ['</s>']

        count = len(tokens)
        total_tokens += count
        sentenceProb = 0
        for i in range(count-1):
            # treat upper and lower case words the same
            word1 = tokens[i].lower()
            word2 = tokens[i+1].lower()
            sentenceProb = sentenceProb + math.log(smoothedBigram(word1, word2, trainTable))
        wordProb = wordProb - sentenceProb

    return (math.exp(wordProb/total_tokens))


def uni_sentiment_classifier(pos_table, neg_table, corpus):
    text_file = open(corpus, 'r')
    lines = text_file.readlines()  
    final_array = []

    pos_tokens = pos_table['SUM'].sum()
    neg_tokens = neg_table['SUM'].sum()

    for line in lines:
        score = 0
        words = line.lower().split()
        for word in words:
            pos_word_count = pos_table.loc[word, 'SUM'] if word in list(pos_table.index.values) else 0
            neg_word_count = neg_table.loc[word, 'SUM'] if word in list(neg_table.index.values) else 0
            score += ( pos_word_count/float(pos_tokens) - neg_word_count/float(neg_tokens) )        
        final_array.append(int(bool(score/len(words) > 0)))
    
    return final_array


def bi_sentiment_classifier(pos_table, neg_table, corpus):
    text_file = open(corpus, 'r')
    lines = text_file.readlines()  
    final_array = []

    pos_bigrams = pos_table.drop(['SUM'], axis=1).values.sum()
    neg_bigrams = neg_table.drop(['SUM'], axis=1).values.sum()

    for line in lines:
        score = 0
        words = ['<s>'] + line.lower().split() + ['</s>']
        for i in range(len(words)-1):
            word1 = words[i]
            word2 = words[i+1]
            pos_count = pos_table.loc[word1, word2] if (word1 in list(pos_table.index.values) and word2 in list(pos_table.columns.values)) else 0
            neg_count = neg_table.loc[word1, word2] if (word1 in list(neg_table.index.values) and word2 in list(neg_table.columns.values)) else 0
            score += ( pos_count/float(pos_bigrams) - neg_count/float(neg_bigrams) )
        final_array.append(int(bool(score/len(words) > 0)))

    return final_array

# Predict the last word
def word_embeddings(filename, model):
    text_file = open(filename, 'r')
    lines = text_file.readlines() 
    correct = []
    predictions = []
    for line in lines:
        words = line.strip().split()
        pos = words[1:3]
        neg = words[:1]
        correct.append(words[3])
        try:
            pred = model.most_similar(positive=pos, negative=neg, topn=1)[0][0]
        except:
            pred = None
        predictions.append(pred)
    return np.array(correct), np.array(predictions)

def cosine_metric(word, filename, model):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    sim = []
    for line in lines:
        w = line.strip().split()
        i = 0
        for i in len(w):
            similar = model.wv.similarity(word, w)
            sim.append(similar)
    sim = Quicksort(sim)
    return sim[:10]

def Quicksort(array, low, high):
    if (len(array)<=1):
        return array
    low = array[0]
    high = array[len(array)-1]
    if low > high:
        p = partition(array, low, high)
    Quicksort(array, high, p+1)
    Quicksort(array, p-1, low)

    
def partition(array):
    pivot = array[high]
    i = low-1
    j = low
    while (j <= high-1):
        if (array[j] <= pivot)
        i+=1
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
    temp = array[i+1]
    array[i+1] = array[high]
    array[high] = temp
    return i+1


if __name__== "__main__":
    start_time = time.time()

    pos_counts = store_counts('SentimentDataset/Train/pos.txt')
    neg_counts = store_counts('SentimentDataset/Train/neg.txt')

    print "---------------- CLASSIFYING SENTIMENT-----------------------"
    pu = uni_sentiment_classifier(pos_counts, neg_counts, 'SentimentDataset/Dev/pos.txt')
    nu = uni_sentiment_classifier(pos_counts, neg_counts, 'SentimentDataset/Dev/neg.txt')
    pb = bi_sentiment_classifier(pos_counts, neg_counts, 'SentimentDataset/Dev/pos.txt')
    nb = bi_sentiment_classifier(pos_counts, neg_counts, 'SentimentDataset/Dev/neg.txt')

    print "----- Evaluating Unigram Sentiment Classifier ------"
    print "Total (Accurately) Predicted Positive Reviews: " + str(sum(pu))
    print "Total Positive Reviews: " + str(len(pu))
    print "Ratio of Accurately Predicted Positive Reviews: " + str( float(sum(pu))/float(len(pu)) )
    
    print "\n"
    print "Total (Accurately) Predicted Negative Reviews: " + str(len(nu) - sum(nu))
    print "Total Negative Reviews: " + str(len(nu))
    print "Ratio of Accurately Predicted Negative Reviews: " + str( float(len(nu) - sum(nu))/float(len(nu)) )
    
    print "\n"
    print "----- Evaluating Bigram Sentiment Classifier -------"
    print "Total (Accurately) Predicted Positive Reviews: " + str(sum(pb))
    print "Total Positive Reviews: " + str(len(pb))
    print "Ratio of Accurately Predicted Positive Reviews: " + str( float(sum(pb))/float(len(pb)) )
    
    print "\n"
    print "Total (Accurately) Predicted Negative Reviews: " + str(len(nb) - sum(nb))
    print "Total Negative Reviews: " + str(len(nb))
    print "Ratio of Accurately Predicted Negative Reviews: " + str( float(len(nb) - sum(nb))/float(len(nb)) )

    print "\n"
    print "Time: " + str(round(time.time()-start_time, 2)) + " seconds"

    #print *****************************************************************************************



