# -*- coding: utf-8 -*-

### Processing and cleaning data
import numpy as np
import pandas as pd
import re
import time
import unicodedata
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from allennlp.commands.elmo import ElmoEmbedder
import random
from GloVe import loadAndCreateGloveModel

DIR = './'

def int_to_cat(y):
    y_cat = []
    for label in y:
        if label == 0:
            y_cat.append('negative')
        elif label == 1:
            y_cat.append('somewhat negative')
        elif label == 2:
            y_cat.append('neutral')
        elif label == 3:
            y_cat.append('somewhat positive')
        elif label == 4:
            y_cat.append('positive')
        else:
            print('Value not between 0 and 4 in the labels !!')
    return y_cat

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def cleaning_text(text, wordnet_lemmatizer): 
    #Input: list of sentences
    #Output: list of sentences
    output = [x.lower() for x in text]# [re.sub('\W+', ' ', strip_accents(x.lower())) for x in text]
    output = [re.sub('  ', ' ', x) for x in output]
    output = [wordnet_lemmatizer.lemmatize(x) for x in output]
    return output

def remove_stopwords(text, stop_words):
    filtered_text = []
    for i in range(len(text)):
        word_tokens = word_tokenize(text[i]) 
        filtered_text.append([w for w in word_tokens if not w in stop_words])
    return filtered_text

def process(filtered_tokens, y, stop_words):    
    filtered_text = [" ".join(x) for x in filtered_tokens]
    unique_text, unique_y = zip(*[x for x in list(set(zip(filtered_text, y))) if len(x[0])>1])
    return list(unique_text), list(unique_y)

def elmo_embedding(unique_text, elmo):
    # unique_text: list of tokens
    X_array = np.zeros((len(unique_text), 256))
    i = 1
    start = time.time()
    for x in unique_text:
        if x == []:
            X_array[i-1,:] = np.zeros((256,))
        else:
            X = elmo.embed_sentence(x)
            X_array[i-1,:] = np.mean(np.mean(X, axis=0), axis=0)
        if i%100 == 0:
            print(str(i)+'/'+str(len(unique_text))+' done in %fs' % (time.time()-start))
            start = time.time()
        i += 1
    return X_array

def embedding(text, model):
    if type(text[0]) == str:
        text = [word_tokenize(x) for x in text]
        print("Tokenized input sentences")
    elif type(text[0]) == list:
        print("Input sentences already tokenized")
    else:
        print("Unknown input type")
        return 0
    d = len(model['hello'])
    embed = np.zeros((len(text), d))
    for i in range(len(text)):
        tokens = text[i]
        vec = np.zeros((d,))
        k = 0
        for x in tokens:
            if x in model.keys():
                vec = vec + model[x]
                k = k + 1
        if k != 0:
            embed[i,:] = vec/k
        else:
            embed[i,:] = vec
    return embed


def svmClassification(X, y):
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = SGDClassifier(loss='log', class_weight="balanced")#MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print('Training score: %.2f' % (accuracy_score(y_train, y_pred_train)))
    print('Testing score: %.2f' % (accuracy_score(y_test, y_pred_test)))
    print('F1 score: %.2f' % (f1_score(y_test, y_pred_test, average='weighted')))
    print('Classification done in %fs' % (time.time()-start))

    cm = confusion_matrix(y_test, y_pred_test)
    sns_heat = sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(set(y_test)), yticklabels=list(set(y_pred_test)))
    fig = sns_heat.get_figure()
    fig.savefig(os.path.join(DIR,'./confusion_matrix.png'))
    print('Figure saved at ', os.path.join(DIR,'./confusion_matrix.png'))
    # return clf

def main(glove, elmo, glove_dim):
    # glove: Boolean. If true, will use glove to embed, train and create a glove_submission.csv
    # elmo: Boolean. If true, will use elmo to embed, train and create an elmo_submission.csv

    ### Import and cleaning training data
    print('Import datas at ', os.path.join(DIR,'train.tsv'))
    df = pd.read_csv(os.path.join(DIR,'train.tsv'), sep='\t')
    sentences = list(df.Phrase)
    y = list(df.Sentiment)
    print("There are %d training sentences." % (len(sentences)))
    assert len(sentences)==len(y)
    print('Start cleaning...')
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    clean_text = cleaning_text(sentences, wordnet_lemmatizer)
    filtered_tokens = remove_stopwords(clean_text, stop_words)
    unique_text, unique_y = process(filtered_tokens, y, stop_words)
    
    ### Import, cleaning and prediction on test data
    print('Import test datas at ', os.path.join(DIR,'test.tsv'))
    df_test = pd.read_csv(os.path.join(DIR,'test.tsv'), sep='\t')
    sentences_test = list(df_test.Phrase)
    print("There are %d testing sentences." % (len(sentences_test)))
    clean_text_test = cleaning_text(sentences_test, wordnet_lemmatizer)
    # clean_text_test = [x if len(x)>1 else '' for x in clean_text_test]
    # filtered_tokens_test = remove_stopwords(clean_text_test, stop_words)
    filtered_tokens_test = [word_tokenize(x) for x in clean_text_test]

    if glove:
        glove_model = loadAndCreateGloveModel(glove_dim)
        X_array = embedding(unique_text, glove_model)
        # svmClassification(X_array, unique_y)  

        ### Training on the whole data
        print('Classification training on the whole dataset...')
        start = time.time()
        clf = SGDClassifier(loss='log', class_weight="balanced")#MultinomialNB()
        clf.fit(X_array, unique_y)
        print('Training done in %fs' % (time.time()-start))

        for i in range(10):
            svmClassification(X_array, unique_y)  

        ### ELMo embedding on testing data + submission generation
        print('Start embedding test...')
        start = time.time()
        X_array_test = embedding(filtered_tokens_test, glove_model)
        print('Embedding test done in %fs' % (time.time()-start))
        y_pred = clf.predict(X_array_test)
        df_test['Sentiment'] = y_pred
        df_test[['PhraseId', 'Sentiment']].to_csv(os.path.join(DIR,'glove_submission'+str(glove_dim)+'.csv'), index=False, sep=',')
        print('Submission created at', os.path.join(DIR,'glove_submission'+str(glove_dim)+'.csv'))

    if elmo:
        ### ELMo embedding on training data
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"

        print("Downloading elmo model...")
        elmo = ElmoEmbedder(options_file, weight_file)
        print("Downloaded.")

        print('Start training embedding...')
        start = time.time()
        try:
            X_array = np.load('X_train.npy')
            print("Existing saved model found !")
        except FileNotFoundError:
            print("No existing saved model found. Begin embedding...")
            X_array = elmo_embedding([x.split() for x in unique_text], elmo)
            np.save('X_train.npy', X_array)
        print('Model loaded in %fs.' % (time.time()-start))
        print("X_train shape: ", X_array.shape)
        ### SVM classification
        print('Start SVM classification on splitting data to get the confusion matrix...')
        svmClassification(X_array, unique_y)   

        ### Training on the whole data
        print('Classification training on the whole dataset...')
        start = time.time()
        clf = SGDClassifier(loss='log', class_weight="balanced")#MultinomialNB()
        clf.fit(X_array, unique_y)
        print('Training done in %fs' % (time.time()-start))

        ### ELMo embedding on testing data + submission generation
        print('Start embedding test...')
        start = time.time()
        try:
            X_array_test = np.load('X_test.npy')
            print("Existing saved model found !")
        except FileNotFoundError:
            print("No existing saved model found. Begin embedding...")
            X_array_test = elmo_embedding(filtered_tokens_test, elmo)
            np.save('X_test.npy', X_array_test)
        print('Embedding test done in %fs' % (time.time()-start))
        y_pred = clf.predict(X_array_test)
        df_test['Sentiment'] = y_pred
        df_test[['PhraseId', 'Sentiment']].to_csv(os.path.join(DIR,'elmo_ssubmission.csv'), index=False, sep=',')
        print('Submission created at', os.path.join(DIR,'elmo_submission.csv'))
    
if __name__ == '__main__':
    main(glove=True, elmo=False, glove_dim=50)
