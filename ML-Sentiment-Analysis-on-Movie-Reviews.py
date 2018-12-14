#################################################################
#
# CMPS 142 - Machine Learning and Data Mining
# Instructor: Snigdha Chaturvedi
#
# Project: Sentiment Analysis on Movie Reviews
# Group Members: Andrew Pang, Pardis Moridian, Jiayi(Jacky) Zhu
#
#################################################################

import nltk, sys, matplotlib, string
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
############################### PROCESSING




# lower-case and tokenize text
def tokenize_text (input_file):
    # to make this compatible with Python2's depreciated U mode
    if sys.version_info[0] == 2:
        f = open(input_file,'rU')
    else:
        f = open(input_file,'r', newline = None)
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    lower_tokens = [token.lower() for token in tokens]
    return lower_tokens


# filter out stopwords. numbers, and punctuation
def filter_out_stopwords_punct (text):
    filtered_text = []
    stop_words = set(stopwords.words('english')) 
    for word in text:
        if word not in stop_words and word not in "!#$%&'()*+,./:;<=>?@[]^_`{|}~" and str(word).isalpha():
            filtered_text.append(word)
    return filtered_text

# finds top n words that occur in a given text
def most_common_words (text, n):
    fdist = FreqDist(text)
    return (fdist.most_common(n))

# stemmer, uses NLTK's own Porter stemmer
def stem_text(text):
    porter = nltk.PorterStemmer()
    return [porter.stem(word) for word in text]

# lemmatizes, again using in-house NLTK fuctions
def lemmatize_text (text):
    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in text]

def applyTransformationToPhrase(dataset):
    for i in range(0, len(dataset)):
        # Tokenize and (uncomment) to apply transformation to Phrase

        tokenized_phrase = nltk.word_tokenize(dataset[i][2][0])
        tokenized_phrase = [token.lower() for token in tokenized_phrase]
        tokenized_phrase = filter_out_stopwords_punct(tokenized_phrase)
        #tokenized_phrase = lemmatize_text(tokenized_phrase)
        #tokenized_phrase = stem_text(tokenized_phrase)
        dataset[i][2] = tokenized_phrase;
    
###################################################### TRAINING




print("Starting...")
start_time = time.time()

# Function to read the input dataset **/
my_data = []
with open ('train.csv') as infile:
    next(infile) #skip first line aka header names
    for instance in infile:
        instance_feature_raw = instance.split(",")
        # Get Phrase ID; Phrase ID is unique!
        PhraseId = int(instance_feature_raw[0])
        # Get Sentence ID
        SentenceId = instance_feature_raw[1]
        # Get Phrase as string
        Phrase = []
        if(len(instance_feature_raw) == 4):
            Phrase.append(instance_feature_raw[2])
        else:
            for x in range(2, len(instance_feature_raw) - 1):
                if(len(Phrase) == 0):
                    Phrase.append(instance_feature_raw[x])
                else:
                    Phrase[0] += instance_feature_raw[x]

        # Get Sentiment
        Sentiment = instance_feature_raw[-1]

        # Store to data set
        my_data.append([PhraseId, int(SentenceId), Phrase, int(Sentiment)])


print("Applying Transformation...")
applyTransformationToPhrase(my_data)
print("Applying Transformation... Done -- " + str(time.time() - start_time) + " secs")


# Remove instances with empty phrases due to filteration of stop words
my_data_final = []
for x in range(0, len(my_data)):
    if(len(my_data[x][2]) > 0):
        my_data_final.append(my_data[x])

print("BoW...")
#Bag of Words
PhraseCorpus = []

numOfInstances = len(my_data_final)
#numOfInstances = 500 # for a smaller training instance set if desired
for x in range(0, numOfInstances):
    PhraseCorpus.append(my_data_final[x][2])
vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
BoW = vectorizer.fit_transform(PhraseCorpus)
# Convert to sparse matrix for faster computation time
BoW = csr_matrix(BoW)
    

print("BoW... Done -- " + str(time.time() - start_time) + " secs")

print("Separating attributes(X) and labels(y) to finalize instances...")
# Separate attributes(X) and labels(y)
y = []
for x in range(0, numOfInstances):
    y.append(my_data_final[x].pop())

y = np.array(y)

print("Separating attributes(X) and labels(y) to finalize instances... Done -- " + str(time.time() - start_time) + " secs")

X_train, X_test, y_train, y_test = train_test_split(BoW, y, test_size = 0.20)

# Train the Algorithm
print("Start Training...")
svclassifier = SVC(kernel='linear', decision_function_shape='ovo')  
svclassifier.fit(X_train, y_train)
print("Start Training... Done -- " + str(time.time() - start_time) + " secs")

y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))




###################################################### TESTING
print("Testing time!...")
# Read in Test file!
my_test_data = []
with open ('testset_1.csv') as infile:
    next(infile) #skip first line aka header names; PhraseId,SentenceId,Phrase
    for instance in infile:
        instance_feature_raw = instance.split(",")
        # Get Phrase ID; Phrase ID is unique!
        PhraseId = int(instance_feature_raw[0])
        # Get Sentence ID
        SentenceId = instance_feature_raw[1]
        # Get Phrase as string
        Phrase = []
        for x in range(2, len(instance_feature_raw)):
            if(len(Phrase) == 0):
                Phrase.append(instance_feature_raw[x])
            else:
                Phrase[0] += instance_feature_raw[x]

        # Store to data set
        my_test_data.append([PhraseId, int(SentenceId), Phrase])


# Create a corpus from test set
PhraseCorpus_test = []
numOfInstances = len(my_test_data)
for x in range(0, numOfInstances):
    PhraseCorpus_test.append(my_test_data[x][2][0])
# Create Bag of Words from test set corpus
BoW_test = vectorizer.transform(PhraseCorpus_test)
BoW_test = csr_matrix(BoW_test)

# Predict!
y_pred_test = svclassifier.predict(BoW_test)


# Save prediction to csv file
with open("predict.csv", "w") as outfile:
    outfile.write("PhraseId, Sentiment\n")
    for x in range(0, len(my_test_data)):
        outfile.write(str(my_test_data[x][0]) + "," + str(y_pred_test[x]) + "\n")


print("\n" + str(time.time() - start_time) + " secs")
