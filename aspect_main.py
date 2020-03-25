from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from joblib import dump, load
# import pickle

import nltk
import time
import random
import argparse

from read_xml import read_xml
from stanfordNLP import sent_to_dep
from aspect_extractor import extractor


# read xml file, only include reviews with one category (single_cate)
f_path = "./data/Restaurants_Train_v2.xml"
reviews = read_xml(f_path, single_cate=True, print_len=False)
test_path = './data/Restaurants_Test_Gold.xml'
test_reviews = read_xml(test_path, single_cate=True, print_len=False)

def p_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--train', action='store_true') #nargs=2, type=str
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return args

args = p_args()

# only need 1000 reviews
random.shuffle(reviews)
reviews = reviews[:1000]

if args.extract:
    # term generation
    start = time.time()
    X = []
    y = []
    cnt = 0
    for row in reviews:
        # try:
        # if cnt < 50:
        cnt += 1
        if cnt % 100 == 0:
            print(cnt/10, "%", round(time.time() - start, 1), "-s")
        review = row[0].lower()
        asp_cat = row[1]
        dep = sent_to_dep(review)
        aspects = extractor(dep)
        # reviews that doesn't have terms won't be in the training set
        if len(aspects) > 0:
            X.append(nltk.FreqDist(aspects))
            y.append(asp_cat[0])

    # print(len(X))

    # transform aspect terms to raw count
    v = DictVectorizer()
    Xs = v.fit_transform(X)

    # dump data
    dump(Xs, open('data/processed_Xs.jbl', 'wb'))
    dump(y, open('data/processed_y.jbl', 'wb'))
    file = open('data/vectorizer.pkl', 'wb')
    dump(v, file)
    file.close()

    end = time.time()
    print("* Timing: ", round(end - start, 1))
    print("* The number of Acspect categories:", len(set(y)))
    print("* So random guess of the model should be:", 1/len(set(y))) # our mdl should beat this score


if args.train:
    # load data
    Xs = load(open('data/processed_Xs.jbl', 'rb'))
    y = load(open('data/processed_y.jbl', 'rb'))

    # data split
    X_train, X_val, y_train, y_val = train_test_split(Xs, y, train_size=0.7, random_state=0)

    # train/valid model
    print("Training 2 models: MultinomialNB, DecisionTree")
    clf1 = MultinomialNB().fit(X_train, y_train)
    clf2 = DecisionTreeClassifier().fit(X_train, y_train)
    acc1 = clf1.score(X_val, y_val)
    acc2 = clf2.score(X_val, y_val)
    print("MultinomialNB Accurary: ", acc1)
    print("DecisionTree Accurary: ", acc2)

    # dump the model with highest accuracy
    if acc1 > acc2:
        print('Dumped MultinomialNB model')
        dump(clf1, open('clf.jbl', 'wb'))
    else:
        print('Dumped DecisionTree model')
        dump(clf2, open('clf.jbl', 'wb'))

if args.test:
    # load mdl
    mdl = load(open('clf.jbl','rb'))
    v = load(open('data/vectorizer.pkl','rb'))
    
    
    # read test data
    test = read_xml('data/Restaurants_Test_Gold.xml', single_cate=True, print_len=False)
    test = test[:100]

    # process data...
    X_test = []
    y_test = []
    cnt = 0
    for row in test:
        # cnt += 1
        # if cnt % 10 == 0:
        review = row[0].lower()
        asp_cat = row[1]
        dep = sent_to_dep(review)
        aspects = extractor(dep)
        # reviews that doesn't have terms won't be in the training set
        if len(aspects) > 0:
            X_test.append(nltk.FreqDist(aspects))
            y_test.append(asp_cat[0])

    # transform aspect terms to raw count
    X_test = v.transform(X_test)
    acc = mdl.score(X_test, y_test)

    print("Test accuracy is: ", acc)