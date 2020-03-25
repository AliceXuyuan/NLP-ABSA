# Aspect-Based-Sentiment-Analysis

## How to run the code

1. Download Stanford Parser files to get access via NLTK
	
	- [Stanford CoreNLP parser (3.9.2 version)](https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip)
	
	- [English language model (3.9.2 English version)](http://nlp.stanford.edu/software/stanford-english-corenlp-2018-10-05-models.jar)

2. Unzip file 1(Stanford CoreNLP parser) and put both of the files under the /config/.

	E.g. 

	File 1: 'config/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'

	File 2: 'config/stanford-english-corenlp-2018-10-05-models.jar'

3. In the command line: $ python aspect_main.py:

	--extract (to extract aspect terms), 

	--train (to train models)

	--test (to use a hold out data set to test model)

4. In the command line: $ python senti_main.py:

	--train_test (to train models and report the accuracy and elapsed time on the testing data)
	
5. In the command line: $ python absa_combined.py
