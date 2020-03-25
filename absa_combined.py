"""
This file is to give the aspect-based sentiment for a given sentence
by combining the categoty model and the sentiment model
"""

import nltk
from joblib import load
from stanfordNLP import sent_to_dep
from aspect_extractor import extractor
from senti_extractor import Features, Vocabularies
from read_xml import read_xml

def sent_sentiment(sent):

    # predict category

    dep = sent_to_dep(sent)
    aspects = extractor(dep)
    raw_count = nltk.FreqDist(aspects)
    v = load(open('data/vectorizer.pkl', 'rb'))
    X = v.transform(raw_count)
    mdl = load(open('clf.jbl', 'rb'))
    category = mdl.predict(X)

    # predict sentiment

    f_path = "./data/Restaurants_Train_v2.xml"
    reviews = read_xml(f_path, single_cate=True, print_len=False)
    voc = Vocabularies(reviews)
    POS = voc.getPOS()
    feature = Features(data=sent, vocabulary=POS, neg=False, file=True)
    feature_set = feature.binary()
    mdl_senti = load(open('classifiers/bayes-m-POS_Binary.jbl', 'rb'))
    sentiment = mdl_senti.predict(feature_set)

    return ("This sentence is expressing a " + str(list(sentiment)[0]) + " attitude about " + str(list(category)[0]))

# ("This sentence is expressing a " + sentiment + " attitude about " + category)




if __name__ == "__main__":
    sent = str(input("Type in a sentence (review)"))
    print(sent_sentiment(sent))


    # predict sentiment
    # feature = getFeatureDict(sent)[3][1]
    # mdl_senti = joblib.load(open('classifiers/bayes-m-POS_Binary.jbl', 'rb'))
    # sentiment = mdl_senti.predict(feature)
