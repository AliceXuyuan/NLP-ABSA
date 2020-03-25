""" This file is to read the xml data:
    we only need reviews and the category of it.
    The formate for each review is as follow:
        [
        "To be completely fair, the only redeeming factor was the food, 
        which was above average, but couldn't make up for all the other 
        deficiencies of Teodora.", 
        ['food', 'anecdotes/miscellaneous']
        ]"""

import xml.dom.minidom
import xml.etree.ElementTree as ET

def read_xml(file_path, print_len=True, single_cate=True):
    with open(file_path) as f:
        file = f.read()
    sentences = ET.fromstring(file)
    sents = sentences.findall("sentence")
    if print_len:
        print('Reviews count: ', len(sents))

    reviews = []
    for i in sents:
        review = i.find('text').text
        # print(len(reviews))
        cates = i.findall("aspectCategories")
        cates_list = [c.findall("aspectCategory") for c in cates][0]
        cate = [c.get("category") for c in cates_list]
        polr = [c.get("polarity") for c in cates_list]
        reviews.append([review, cate, polr])
        
        if single_cate:
            res = []
            for r in reviews:
                if len(r[1]) == 1:
                    res.append(r) 
        else:
            res = reviews
        
    return res
    

if __name__ == "__main__":
    f_path = "./data/Restaurants_Train_v2.xml"

    reviews_single = read_xml(f_path, single_cate=True)
    print('# of Single Cate Reviews: ',len(reviews_single))

    reviews_multi = read_xml(f_path, single_cate=False)
    print('# of Multi Cate Reviews: ',len(reviews_multi))
    print('Data format example:\n ', reviews_multi[1])