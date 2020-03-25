from nltk.corpus import sentiwordnet as swn
from nltk.stem.wordnet import WordNetLemmatizer

from stanfordNLP import sent_to_dep


verb = ["VBD", "VB", "VBG", "VBN","VBP", "VBZ"]
noun = ["NN", "NNS", "NNP", "NNPS"]
adverb = ["RB", "RBR", "RBS"]
adjective = ["JJ", "JJR", "JJS"]
auxiliary_verb = ["be", "am", "are", "is", "was", "being", "can", "could", "do", "did", "does", "doing", "have", "had",
         "has", "having", "may", "might", "might", "must", "shall", "should", "will", "'ve", "n't", "were"]
asdict={}
# lem=WordNetLemmatizer()


def getVmodorXcomp(words={}, word=""):
    for key in words[word].keys():
        if((key == "vmod" or key == "xcomp")):
            tmp =  words[word][key]
            if(words[tmp]["pos_tag"] in (adverb or adjective)):
                return tmp
    return None


def hasPropositionalNoun(words={}, word=""):
    for key in words[word].keys():
        if ("prep" in key):
            tmp = words[word][key]
            if(words[tmp]["pos_tag"] in noun):
                return tmp
    return None


def getAuxiliary(words={}):
    for word in words.keys():
        for key in words[word]:
            if("aux" == key):
                return True
    return False


def getDependency(words={}, word=""):
    for key in words[word].keys():
        try:
            if(key != "xcomp"):
                continue
            tmp = words[word][key]
            if(words[tmp]["pos_tag"] in verb):
                return tmp
        except:
            continue
    return None


def findNsubj(words={}):
    for key in words.keys():
        if "nsubj" in words[key]:
            return True
    return False


def checkModifiers(words = {}, word=""):
    try:
        tmp = words[word]["amod"]
        return wordInSentic(tmp)
    except:
        pass
    try:
        tmp = words[word]["advmod"]
        return wordInSentic(tmp)
    except:
        pass
    return False


def getProposition(words = {}, word=""):
    if word=="" or len(words)==0:
        return None
    for i in words[word].keys():
        if "prep" in i:
            return i
    return None


def getAdverbOrAdjective(words = {}, word=""):
    for key in words[word].keys():
        tmp = words[word][key]
        try:
            if(key == "advmod"):
                return True
            elif(words[tmp]["pos_tag"] in (adverb + adjective)):
                return True
        except:
            continue
    return False


def getNounConnectedByAny(words={}, word=""):
    if (word=="" or len(words)==0):
        return None
    for dep in words[word].keys():
        try:
            if (words[words[word][dep]]["pos_tag"] in noun):
                return words[word][dep]
        except:
            continue


def isNoun(words={}, word=""):
    return (words[word]["pos_tag"] in noun)


def wordInSentic(w=""):
    l = list(swn.senti_synsets(w))
    if len(l) > 0:
        return True
    else: # if: word in sentic xml
        return False


def isInOpinionlexicon(word = ""):
    return wordInSentic(word)


def isNounSubject(words = {}):
    if (len(words) == 0):
        return None
    for word in words.keys():
        if "nsubj" in words[word]:
            return True
    return False


def extractor(words):
    aspect_terms = []
    has_auxiliary = getAuxiliary(words)
    hasNsubj = findNsubj(words)

    # General rule - 1: having a subject-verb
    if hasNsubj:
        for word in set(words.keys()):
            if not words[word].__contains__("nsubj"):
                continue
            # Point 1
            if words[word]["pos_tag"] in verb and checkModifiers(words, word):
                aspect_terms.append(word)
            # Point 2
            elif not has_auxiliary and getAdverbOrAdjective(words, word):
                try:
                    if (words[word].__contains__("nsubj") and not ("DT" in words[words[word]["nsubj"]]["pos_tag"])):
                        aspect_terms.append(words[word]["nsubj"])
                        aspect_terms.append(word)
                except:
                    pass
                    # print()
            # Point 3
            elif not has_auxiliary and words[word].__contains__("dobj") and isNoun(words, words[word]["dobj"]):
                if not wordInSentic(words[word]["dobj"]):
                    aspect_terms.append(words[word]["dobj"])
                else:
                    word1 = getNounConnectedByAny(words, words[word]["dobj"])
                    if (word1):
                        aspect_terms.append(word1)
            # Point 4
            elif not has_auxiliary and (words[word].__contains__("xcomp")):
                xcomp = words[word]["xcomp"]
                word1 = getNounConnectedByAny(words, xcomp)
                if (word1):
                    aspect_terms.append(word1)
            # Point 5 & 6 & 7
            elif "cop" in words[word]:
                dep = getDependency(words, word)
                if wordInSentic(word) and words[word]["pos_tag"] in noun:
                    # print("In 5", word)
                    aspect_terms.append(word)
                try:
                    # if "nsubj" in words[lem.lemmatize(word)] \
                    #     and not ("DT" in words[lem.lemmatize(words[lem.lemmatize(word)]["nsubj"])]["pos_tag"] \
                    # or "PRP" in words[lem.lemmatize(words[lem.lemmatize(word)]["nsubj"])]["pos_tag"]):
                        # print("In 6", words[word]["nsubj"])
                    if "nsubj" in words[word] \
                        and not ("DT" in words[words[word]["nsubj"]]["pos_tag"] \
                    or "PRP" in words[words[word]["nsubj"]]["pos_tag"]):
                        aspect_terms.append(words[word]["nsubj"])
                except KeyError:
                    pass
                if dep:
                    # print("In 7", dep)
                    aspect_terms.append(dep)
    
    # General rule - 2: NOT having a subject-verb
        else:
            for word in words.keys():
                prepN = hasPropositionalNoun(words, word)
                tmp = getVmodorXcomp(words, word)
                if tmp and wordInSentic(tmp):
                    # print("In 8", word)
                    aspect_terms.append(word)
                elif prepN:
                    # print("In 9")
                    if "appos" in words[word]:
                        # print(words[word]["appos"])
                        aspect_terms.append(words[word]["appos"])
                    # print(prepN)
                    aspect_terms.append(prepN)
                elif "dobj" in words[word]:
                    try:
                        # print("In 10")
                        tmp1 = words[words[word]["dobj"]]["pos_tag"]
                        if not ("DT" in tmp1) or ("PRP" in tmp1):
                            # print(words[word]["dobj"])
                            aspect_terms.append(words[word]["dobj"])
                    except KeyError:
                        pass

    return aspect_terms


if __name__ == "__main__":
    # sample
    # sent = "The food is uniformly exceptional, with a very capable kitchen which 
    #       will proudly whip up whatever you feel like eating, whether it's on the 
    #       menu or not."

    # Myaspect = extractor(
        # {'feel': {'pos_tag': 'VBP', 'dobj': 'whatever', 'nsubj': 'you', 
        #           'ccomp': 'menu', 'nmod': 'eating'}, 
        # 'menu': {'pos_tag': 'NN', 'cc': 'or', 'conj': 'not', 'nsubj': 'it', 
        #           'mark': 'whether', 'cop': "'s", 'case': 'on', 'det': 'the'}, 
        # 'kitchen': {'pos_tag': 'NN', 'det': 'a', 'advmod': 'exceptional', 
        #           'amod': 'capable', 'acl:relcl': 'whip', 'case': 'with', 
        #           'cop': 'is', 'nsubj': 'food'}, 
        # 'food': {'pos_tag': 'NN', 'det': 'The'}, 
        # 'whip': {'pos_tag': 'VB', 'advmod': 'proudly', 'nsubj': 'which', 
        #           'compound:prt': 'up', 'ccomp': 'feel', 'aux': 'will'}, 
        # 'eating': {'pos_tag': 'NN', 'case': 'like'}, 
        # 'exceptional': {'pos_tag': 'RB', 'advmod': 'uniformly'}, 
        # 'capable': {'pos_tag': 'JJ', 'advmod': 'very'}})

    sent = "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not."
    dep = sent_to_dep(sent)
    print(extractor(dep))