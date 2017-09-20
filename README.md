# CS4740_P1

#### Create a folder 'word_vectors' in the 'Project1' folder
#### Download the word2vec model from the 'word2vec' link in https://piazza.com/class/j6qllgvbygx4mf?cid=250
#### Download the glove model from the link in https://piazza.com/class/j6qllgvbygx4mf?cid=248
#### Once downloaded, move both files to word_vectors, and double-click the glove model (but not the GoogleNews model)

#### In terminal, cd into the word_vectors folder and run these two commands:

gunzip -k GoogleNews-vectors-negative300.bin.gz 
python -m gensim.scripts.glove2word2vec --input glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt

#### Note: to use, you might have to install the gensim package i.e. `pip install gensim` in terminal in order to use them in python
