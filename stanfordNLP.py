import nltk
from nltk.parse.stanford import StanfordDependencyParser 
import warnings


# ignore StanfordNLP deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# setup StanforNLP parser
path_to_jar = 'config/stanford-english-corenlp-2018-10-05-models.jar'
path_to_models_jar = 'config/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'


def sent_to_dep(sent):
    """return a dictionary containing governor words and their dependency"""
    # set up StanfordNLP parser
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar,
    path_to_models_jar=path_to_models_jar)
    # parse a sentence and get the dependency
    result = dependency_parser.raw_parse(sent)
    dep = result.__next__()
    output = set(dep.triples())
    dic = {}
    # adjust the pattern
    for element in output:
        governor = element[0][0]
        dep = element[1]
        dependent = element[2][0]
        pos_tag = element[0][1]
        if governor not in set(dic.keys()):
            dic[governor] = {'pos_tag': pos_tag, dep: dependent}
        else:
            dic[governor][dep] = dependent
    # generate pos_tag for words without pos_tag
    tokens = nltk.word_tokenize(sent)
    pos_tag = nltk.pos_tag(tokens)
    for t in pos_tag:
        word = t[0]
        tag = t[1]
        if word not in dic.keys():
            dic[word] = {'pos_tag': tag}
    return dic


if __name__ == "__main__":

    sent = "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not."
    print(sent)
    print("Dependency: ", sent_to_dep(sent))