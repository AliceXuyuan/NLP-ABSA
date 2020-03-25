#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:48:37 2019

@author: tianyizhou
"""


from senti_extractor import model
from senti_extractor import Vocabularies
from senti_extractor import Features
from senti_extractor import getFeatureDict

from read_xml import read_xml

import argparse
import warnings
warnings.filterwarnings("ignore")

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test", dest='train_test', action='store_true',
                        help='train and test the model')
#    parser.add_argument("--run", dest='run', nargs=2,
#                        metavar=('model_name', 'file_name'),
#                        help='run the chosen model on text file')
    args = parser.parse_args()
    return args


# Call the Parser function
args = Parser()

if args.train_test:
    # Import restaurant data
    # training data
    f_path = "./data/Restaurants_Train_v2.xml"
    train = read_xml(f_path, print_len=False, single_cate=True)
    train_target = [s[2] for s in train]    # testing data
    
    # testing data
    f_path = "./data/Restaurants_Test_Gold.xml"
    test = read_xml(f_path, print_len=False, single_cate=True)
    test_target = [s[2] for s in test]   
    
    # Generate vocabularies for creating features for models
    # Use training data
    voc = Vocabularies(train)
    Raw = voc.getRaw()
    Raw_neg = voc.getRaw_neg()
    POS = voc.getPOS()
    POS_neg = voc.getPOS_neg()
    POS_DT = voc.getPOSnDT()
    POS_DT_neg = voc.getPOSnDT_neg()
    swn = voc.getSentiwordnet()
    swn_neg = voc.getSwn_neg()
    mpqa = voc.getMPQA()
    mpqa_neg = voc.getMPQA_neg()
    voc_list = [swn, mpqa, Raw, POS, POS_DT, swn_neg, mpqa_neg, Raw_neg, POS_neg, POS_DT_neg]
    
    # Generate features dictionary
    features_train = getFeatureDict(train,voc_list)
    features_test = getFeatureDict(test,voc_list)
    
    # Create a mapping dictionary for vocabularies and fetures
    voc_names = {1: 'swn', 2: 'mpqa', 3: 'Raw', 4: 'POS', 5: 'POS_DT',
             6: 'swn_neg', 7: 'mpqa_neg', 8: 'Raw_neg', 9: 'POS_neg', 10: 'POS_DT_neg'}
    feature_names = {1: 'Count', 2: 'Binary'}

    classifiers = ['bayes-m','bayes-b','tree','svm',
                   'linear-svm','neuro network']
    for i in classifiers:
        model(i, voc_names, feature_names, features_train,
              train_target, features_test, test_target)

# if args.run:

#     i = int(input("""
#           Choose a model:
#           1 - SentiWordNet words
#           2 - Subjectivity Lexicon words
#           3 - all words raw
#           4 - POS
#           5 - POS with Distributional Thesaurus
#           6 - SentiWordNet words with negation
#           7 - Subjectivity Lexicon words with negation
#           8 - all words raw with negation
#           9 - POS with negation
#           10 - POS with Distributional Thesaurus with negation
#           Type a number:\n"""))
          
#     model_name, file_name = args.run
    
#     #Unseen test data
#     f_path = "./data/Restaurants_Test_Gold.xml"
#     with open(f_path) as f:
#         file = f.read()
#     test = read_xml(file, single_cate=True)
#     test_target = [s[2] for s in test]   


#     # predict
#     # load the model
#     with open('classifier/{}-{}.jbl'.format(model_name,
#                                              features_dict[i]), 'rb') as md:
#         clf = pickle.load(md)
#     # load the vector
#     with open('vector/{}.jbl'.format(features_dict[i]), 'rb') as vt:
#         vector = pickle.load(vt)
#         array = vector.transform(file)
#     print(clf.predict(array))