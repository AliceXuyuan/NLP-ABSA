#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:51:43 2019

@author: tianyizhou shengchenfu
"""
from read_xml import read_xml

import nltk
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.treebank import TreebankWordDetokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import os
import re
import time
import joblib
import pandas as pd


# Single Aspect Polarity
# Rules Based Models
class Vocabularies(object):
    # Create vocabularies for generating features
    def __init__(self, reviews):
        self.reviews = reviews
        self.sentences_raw = [i[0] for i in self.reviews]
        self.sentences = [i.lower() for i in self.sentences_raw]
        self.tokens = [word_tokenize(sentence) for sentence in self.sentences]
        self.pos_tags = [pos_tag(token, tagset='universal') for token in self.tokens]
        self.words = set(sum(self.tokens, []))

    def getRaw(self):
        # Create raw words without stopwords based on all reviews
        stopwords = nltk.corpus.stopwords.words('english')
        self.raw = set(w for w in self.words if w not in stopwords)
        return self.raw

    def getRaw_neg(self):
        # Add negated form words to previous raw words
        raw_neg = self.raw.copy()
        for w in self.raw:
            negated_w = 'not_' + w
            raw_neg.add(negated_w)
        return raw_neg

    def getPOS(self):
        # Rule 1: Part of Speech is adj, adv or verb
        # Return a set of 2163/2159 words
        self.sentiment_terms = set()
        for i in self.pos_tags:
            for j in i:
                if j[1] in ['ADJ', 'ADV', 'VERB']:
                    self.sentiment_terms.add(j[0])
        return self.sentiment_terms

    def getPOS_neg(self):
        # Add negated form words to previous sentiment terms
        s_terms_neg = self.sentiment_terms.copy()
        for w in self.sentiment_terms:
            negated_w = 'not_' + w
            s_terms_neg.update(negated_w)
        return s_terms_neg

    def getPOSnDT(self):
        # Rule 2: add Distributional Thesaurus into POS vocabularies
        # Use the top 5 DT expansions of current token
        # Return a set of 8157 words
        self.sentiment_terms_DT = self.sentiment_terms.copy()
        for i in self.sentiment_terms:
            synonyms = set()
            for syn in wordnet.synsets(i)[:2]:
                for w in syn.lemmas():
                    synonyms.add(w.name())
            self.sentiment_terms_DT.update(synonyms)
        return self.sentiment_terms_DT

    def getPOSnDT_neg(self):
        # Add negated form words to
        # previous sentiment terms with distributional thesaurus
        neg_DT_terms = self.sentiment_terms_DT.copy()
        for w in self.sentiment_terms_DT:
            negated_w = 'not_' + w
            neg_DT_terms.update(negated_w)
        return neg_DT_terms

    # Feature 3: Count SentiWordNet words with positive or negative over 0.5
    def getSentiwordnet(self):
        # both in reviews and have pos/neg over 0.5 in SentiWordNet
        self.common_words = set()
        for w in self.words:
            if list(swn.senti_synsets(w)):
                senti = list(swn.senti_synsets(w))[0]
                polarity = [senti.pos_score(), senti.neg_score()]
                if any(polarity) > 0.5:
                    self.common_words.add(w)
        return self.common_words

    def getSwn_neg(self):
        # Add negated form words to previous raw words
        swn_neg = self.common_words.copy()
        for w in self.common_words:
            negated_w = 'not_' + w
            swn_neg.add(negated_w)
        return swn_neg

    def getMPQA(self):
        # Generate the words list from MPQA
        with open('data/''subjclueslen1-HLTEMNLP05.tff', 'r') as f:
            mpqa = f.read()
        mpqa_words = set(re.findall(r"(?<=word1=)\w+", mpqa))
        self.common_words_m = set([w for w in mpqa_words if w in self.words])
        return self.common_words_m

    def getMPQA_neg(self):
        mpqa_neg = self.common_words.copy()
        for w in self.common_words:
            negated_w = 'not_' + w
            mpqa_neg.add(negated_w)
        return mpqa_neg


class Features(object):
    def __init__(self, data, vocabulary, neg=False, file=False):
        # file argument by default is False, when it is true it means we take
        # unseen data file for test
        if file is True:
            self.sentences = [data.lower()] 
        else:
            self.reviews = data
            self.sentences_raw = [i[0] for i in self.reviews]
            self.sentences = [i.lower() for i in self.sentences_raw]
        self.tokens = [word_tokenize(sentence) for sentence in self.sentences]
        self.vectorizer = CountVectorizer(vocabulary=vocabulary)
        self.neg = neg

    def negation(self):
        # Conduct negation on sentences
        neg_tokens = ["n't", "never", "nothing", "nowhere", "none", "not", "no"]
        delims = ",.!?;:"
        negation = False
        self.neg_sentences = len(self.sentences)*[None]
        d = TreebankWordDetokenizer()
        for i in range(len(self.sentences)):
            tokens = self.tokens[i]
            negated_tokens = []
            for w in tokens:
                negated_w = 'not_' + w if negation else w
                negated_tokens.append(negated_w)
                if w in neg_tokens:
                    negation = True
                if w in delims:
                    negation = False
            self.neg_sentences[i] = d.detokenize(negated_tokens)
        return self.neg_sentences

    def count(self):
        sentences = self.negation() if self.neg else self.sentences
        count = self.vectorizer.fit_transform(sentences)
        self.count_array = count.toarray()
        return self.count_array

    def binary(self):
        count_array = self.count()
        return binarize(count_array)


# Prepare vocabularies for generating features
def getFeatureDict(data,voc_list):
    # Prepare features for training models
    # A dictionary with 0-5 as keys
    # Each key has 2 feature arrays
    featureDict = {}
    # Generate not negated pairs and negated pairs of vocabulary and feature type
    for i in range(len(voc_list)):
        TwoFeatures = []
        neg = False if i < 5 else True  # swntest
        features = Features(data=data, vocabulary=voc_list[i], neg=neg)
        TwoFeatures.append(features.count())
        TwoFeatures.append(features.binary())
        featureDict.update({i: TwoFeatures})
    return featureDict


def model(classifier, voc_names, feature_names,
          features_train, train_target, features_test, test_target):
    # Create the classifiers directory to store models
    if not os.path.isdir("classifiers/"):
        os.mkdir("classifiers/")
    # create a df for output elapsed_time and accuracy
    row_name = list(voc_names.values())
    column_name = ['acc_count','acc_binary','time_count','time_binary']
    df = pd.DataFrame(columns=column_name, index=row_name)
    # Train and run models
    for i in range(len(voc_names)):
        for j in range(len(feature_names)):
            start = time.time()
            feature = features_train[i][j]

            if classifier == "bayes-m":
                c = MultinomialNB()
                c.fit(feature, train_target)
            if classifier == 'bayes-b':
                c = BernoulliNB()
                c.fit(feature, train_target)
            if classifier == "tree":
                c = DecisionTreeClassifier()
                c.fit(feature, train_target)
            if classifier == 'svm':
                c = svm.SVC()
                c.fit(feature, train_target)
            if classifier == 'linear-svm':
                c = svm.LinearSVC()
                c.fit(feature, train_target)
            if classifier == 'neuro network':
                c = MLPClassifier(solver='adam', alpha=1e-5,
                                  hidden_layer_sizes=(1000,), random_state=1)
                c.fit(feature, train_target)

            # Store models by classifier-feature
            feature_name = voc_names[i+1]+'_'+feature_names[j+1]
            joblib.dump(c, 'classifiers/{}-{}.jbl'.
                        format(classifier, feature_name))
            print('Creating {} classifiers in classifiers/{}-{}.jbl'.
                  format(classifier, classifier, feature_name))

            # Elapsed time
            end = time.time()
            elapsed_time = round(end - start, 2)
            print('Elapsed time: {}s'.format(elapsed_time))

            # Accuracy on testing data
            ypred = c.predict(features_test[i][j])
            accuracy = round(accuracy_score(test_target, ypred), 2)
            print('Accuracy: {}'.format(accuracy))
            df.set_value(row_name[i],column_name[j], accuracy)
            df.set_value(row_name[i],column_name[j+2], elapsed_time)
    return df


if __name__ == "__main__":

    # Output result to Excel
       with pd.ExcelWriter('senti_result.xlsx', engine='xlsxwriter') as writer:
            #classifiers = ['bayes-m','bayes-b','tree','svm','linear-svm','neuro network']
            classifiers = ['bayes-m','bayes-b','tree','svm',
                           'linear-svm','neuro network']
            for i in classifiers:
               result = model(i)
               result.to_excel(writer, sheet_name=i, header=True)
       writer.save()
        


