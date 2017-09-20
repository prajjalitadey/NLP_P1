from collections import defaultdict
from pprint import pprint
import pandas as pd
from random import randint, randrange


# create bigram table
def store_counts(filename):
    text_file = open(filename, 'r')
    lines = text_file.readlines()
    types = defaultdict(lambda: defaultdict(int))

    #instantiate seen word set
    seen = set()

    for line in lines:
        #Standardize the contractions ('t is a separate word)
        tokens = line.replace(" n't", "n 't")
        #Get rid of hyphens
        tokens = line.replace('-', ' ')
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
    table = pd.DataFrame(types).T.fillna(0).applymap(lambda x: int(x))
    # add totals
    table['SUM'] = table.sum(axis=1)
    table.loc['</s>', 'SUM'] = int(table.loc['<s>', 'SUM'])
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


# return the bigram P(word2|word1) for the given table of counts
def bigram(word1, word2, table):
    try:
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
def biPerplexity(trainTable, filename):
    text_file = open(filename, 'r')
    lines = text_file.readlines()
    types = defaultdict(lambda: defaultdict(int))


    wordProb = 0

    for line in lines:
        #Standardize the contractions ('t is a separate word)
        tokens = line.replace(" n't", "n 't")
        #Get rid of hyphens
        tokens = line.replace('-', ' ')
        tokens = ['<s>'] + line.split() + ['</s>']

        count = len(tokens)

        sentenceProb = 0

        for i in range(count-1):
            # treat upper and lower case words the same
            word1 = tokens[i].lower()
            word2 = tokens[i+1].lower()

            sentenceProb = sentenceProb + math.log(smoothedBigram(word1, word2, trainTable))

        wordProb = wordProb - sentenceProb

    return (math.exp(wordProb/trainTable['SUM'].sum()))


def sentiment_classification(corpus):
    text_file = open(corpus, 'r')
    lines = text_file.readlines()

    pos_counts = store_counts('pos.txt')
    neg_counts = store_counts('neg.txt')
    final_array = []
    # pos = 1; neg = 0
    for line in lines:
        pos_perplexity = uniPerplexity(pos_counts, line)
        neg_perplexity = uniPerplexity(neg_counts, line)
        final_array.append(bool(pos_perplexity < neg_perplexity))
    return final_array



if __name__== "__main__":
    pos_counts = store_counts('pos.txt')
    neg_counts = store_counts('neg.txt')

    print "---------------- CALCULATING POSITIVE UNIGRAM GENERATED SENTENCES -------------"
    for i in range(5):
        print str(i+1) + '. ' + rsgUnigram(pos_counts)
    print '\n'

    print "---------------- CALCULATING POSITIVE BIGRAM GENERATED SENTENCES -------------"
    for i in range(5):
        print str(i+1) + '. ' + bigram_sentence_generator(pos_counts)
    print '\n'

    print "---------------- CALCULATING NEGATIVE UNIGRAM GENERATED SENTENCES -------------"
    for i in range(5):
        print str(i+1) + '. ' + rsgUnigram(neg_counts)
    print '\n'

    print "---------------- CALCULATING NEGATIVE BIGRAM GENERATED SENTENCES -------------"
    for i in range(5):
        print str(i+1) + '. ' + bigram_sentence_generator(neg_counts)
    print '\n'
