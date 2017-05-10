from os import listdir
from os.path import isfile, isdir, join
import csv
from enchant.checker import SpellChecker
import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tag import PerceptronTagger
from nltk.corpus import cess_esp as cess
from nltk.corpus import floresta
from sklearn.svm import LinearSVC, SVC
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
import time
import string
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from sklearn.preprocessing import Normalizer
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import codecs
import gc
import argparse
import nltk
from itertools import groupby
#import gensim
#from gensim.models.doc2vec import TaggedDocument
#from experimentation import compress


import resource
rsrc = resource.RLIMIT_AS
soft, hard = resource.getrlimit(rsrc)
resource.setrlimit(rsrc, (13500000000, hard)) #limit allowed Python memory usage to 13GB

start_time = time.time()
lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
chkr = SpellChecker("en_US")


def generate_output(path, author_id, lang, variety, gender):
    root = ET.Element("author")
    root.set('id', author_id)
    root.set('lang', lang)
    root.set('variety', variety)
    root.set('gender', gender)
    tree = ET.ElementTree(root)
    tree.write(join(path, author_id + ".xml"))


#read PAN 2016 tweet corpus from xml and extract tweet texts, variety and gender of the author. Write to csv for later use.
def readPANcorpus(path, lang, test=False):
    path = join(path, lang)
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('xml')]
    data = [['id', 'gender', 'variety', 'text']]
    cntRT = 0
    cntTweet = 0
    gender_dict = {}
    variety_dict = {}
    truth_file = join(path, 'truth.txt')
    if isfile(truth_file):
        for line in open(truth_file):
            l = line.split(':::')
            gender_dict[l[0]] = l[1].strip()
            variety_dict[l[0]] = l[2].strip()

    for file in files:
        name = file.split('.')[0].split('/')[-1]
        try:
            tree = ET.parse(file)
        except:
            continue
        root = tree.getroot()
        if isfile(truth_file):

            gender = gender_dict[name]
            variety = variety_dict[name]
        else:
            gender = 'undefined'
            variety = 'undefined'
        concatenated_text = ""
        for document in root.iter('document'):
            cntTweet += 1
            if document.text:
                txt = beautify(document.text)
                if lang == 'en' and find_garbage_rate(txt, chkr) > 0.9:
                        pass
                elif document.text.startswith("RT "):
                    cntRT += 1
                else:
                    tweet = txt.replace("\n", " ").replace("\t", " ")
                    concatenated_text += tweet + "\n"
        #remove empty strings
        if concatenated_text:
            data.append([name, gender, variety, concatenated_text.strip()])
        else:
            print(name)
    #print(cntRT)
    print('Number of Tweets: ', cntTweet)
    if not test:
        #write to csv file
        with open('csv_files/PAN_data_' + lang + '.csv', 'w') as fp:
            a = csv.writer(fp, delimiter='\t')
            a.writerows(data)
    else:
        headers = data.pop(0)
        data = pd.DataFrame(data, columns=headers)
        return data


#read different word lists and return a set of words
def read_wordList(file):
    with open(file, 'r') as f:
        return set([word.split('\t')[0].strip() for word in f if word])


#remove html tags, used in PAN corpora
def beautify(text):
    return BeautifulSoup(text, 'html.parser').get_text()


def lemmatize(text):
    text = word_tokenize(text)
    l = []
    for w in text:
        #lemmagen crashes python on very long words for some reason
        if len(w) < 20:
            try:
                w = lemmatizer.lemmatize(w)
            except:
                print(w)
        l.append(w)
    # convert list back to string
    return " ".join(l)


def tokenize_n_character(text):
    return text.split()


def remove_punctuation(text):
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text


def remove_stopwords(text, lang):
    if lang == 'es':
        stops = set(stopwords.words("spanish"))
    elif lang == 'en':
        stops = set(stopwords.words("english"))
    elif lang == 'pt':
        stops = set(stopwords.words("portuguese"))
    else:
        return text
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)


def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token):
    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def tag(tagger, text, sent_tokenize):
    #for arab we dont do POS tagging
    if not sent_tokenize:
        return text
    #tokenize with nltk default tokenizer
    tokens = sent_tokenize.tokenize(str(text))
    #use average perceptron tagger
    tokens = [word_tokenize(token) for token in tokens]
    text = tagger.tag_sents(tokens)
    return " ".join(tag for sent in text for word, tag in sent)

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t

def get_emojis(path):
    emoji_dict = {}
    df_emojis = pd.read_csv(path, encoding="utf-8", delimiter=",")
    for index, row in df_emojis.iterrows():
        occurrences = row['Occurrences']
        pos = (row['Positive'] + 1) / (occurrences + 3)
        neg = (row['Negative'] + 1) / (occurrences + 3)
        sent = pos - neg
        emoji_dict[row['Emoji']] = sent
    return emoji_dict


def countCharacterFlooding(text):
    text = ''.join(text.split())
    groups = groupby(text)
    cnt = 0
    for label, group in groups:
        char_cnt = sum(1 for _ in group)
        if char_cnt > 2:
            cnt += 1
    return cnt

#count words in tweet that are in a specific word list and return frequency
def countWords(wordList, text):
    cnt = 0
    length = len(text.split())
    for word in text.split():
        if word in wordList:
            cnt +=1
    if length == 0:
        return 0
    return cnt/length

#count specific characters
def count_patterns(text, list):
    cnt=0
    length = len(text)
    for pattern in list:
        cnt += text.count(pattern)
    if length == 0:
        return 0
    return cnt/length

#get sentiment according to emojis
def get_sentiment(text, emoji_dict):
    sentiment = 0
    list = emoji_dict.keys()
    for pattern in list:
        text_cnt = text.count(pattern)
        sentiment += emoji_dict[pattern] * text_cnt
    return sentiment


#count specific pos tags and return frequency
def count_pos(pos_sequence, pos_list):
    cnt = 0
    for pos_tag in pos_sequence.split():
        for pos in pos_list:
            if pos_tag == pos:
                cnt += 1
    return cnt/len(pos_sequence.split())



#used for feature evaluation
def most_informative_feature_for_class(vectorizer, classifier, n=40):
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
    class_labels = classifier.classes_

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


#find how many words in a document are spelled incorrectly
def find_garbage_rate(text, chkr):
    text = remove_hashtags(text, '')
    text = remove_mentions(text, '')
    text = remove_url(text, '')
    text = remove_punctuation(text)
    chkr.set_text(text)
    text_len = len(text.split())
    if text_len > 0:
        error_list = [err.word for err in chkr]
        error_rate = len(error_list)/len(word_tokenize(text))
        if error_rate > 0.9:
            pass
        return error_rate
    else:
        return 1

#used in count error feature
def count_error(text, chkr):
    text = text.split('\n')
    num_tweets = len(text)
    common_error = 0
    for tweet in text:
        chkr.set_text(tweet)
        tweet_len = len(word_tokenize(tweet))
        if tweet_len > 0:
            error_list = [err.word for err in chkr]
            error_rate = len(error_list)/tweet_len
        else:
            error_rate = 0
        common_error += error_rate
    return common_error/num_tweets


def get_affix(text):
    return " ".join([word[-4:] if len(word) >= 4 else word for word in text.split()])


def get_prefix(text):
    return " ".join([word[0:4] for word in text.split() if len(word) > 4])


def affix_punct(text):
    punct = '!"$%&()*+,-./:;<=>?[\]^_`{|}~'
    ngrams = []
    for i, character in enumerate(text[0:-2]):
        ngram = text[i:i+3]
        if ngram[0]  in punct:
            for p in punct:
                if p in ngram[1:]:
                    break
            else:
                ngrams.append(ngram)
    return "###".join(ngrams)

def affix_punct_tokenize(text):
    tokens = text.split('###')
    return tokens

'''def makeDocVecFeatures(text, model, sent_tokenizer, num_features=300):
    tokens = sent_tokenizer.tokenize(str(text))
    tokens = [word for sent in tokens for word in word_tokenize(sent)]
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in tokens:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def create_word2vec_model(df_data, sent_tokenizer, lang, num_features=300):
    if not isfile('w2v_' + lang + '.model'):
        all_docs = df_data['text_clean'].tolist()
        text = " ".join(all_docs)
        tokens = sent_tokenizer.tokenize(str(text))
        tokens = [word_tokenize(token) for token in tokens]
        model = gensim.models.Word2Vec(tokens, size=num_features, window=20, min_count=40, workers=11, sample = 1e-3)
        model.save('doc2vec_' + lang + '.model')
    else:
        model = gensim.models.Doc2Vec.load('doc2vec_' + lang + '.model')
    return model'''


def get_ngrams(text):
    ngrams = []
    for word in text.split():
        if len(word) > 4:
            for i in range(len(word) - 3):
                ngrams.append(word[i:i + 4])
        else :
            ngrams.append(word)
    print(ngrams)
    return " ".join(ngrams)


#fit and transform text features, used in scikit Feature union
class text_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]


#fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['text', 'pos_tag', 'no_punctuation', 'no_stopwords', 'text_clean', 'affixes', 'affix_punct']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        scaler = preprocessing.MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)


# fit and transform w2v features, used in scikit Feature union
class w2v_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        hd_searches = data_dict[self.key]
        hd_searches = [x.tolist() for x in hd_searches]
        #hd_searches = hd_searches.values
        scaler = preprocessing.MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)


#needed so xgboost doesn't crash
class CSCTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.tocsc()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


#preprocess and tag data and write it to csv for later use
def preprocess(df_data, lang, perceptron_tagger, sent_tokenizer, test=False):
    print('clean text')
    df_data['text_clean_r'] = df_data['text'].map(lambda x: remove_hashtags(x, '#HASHTAG'))
    df_data['text_clean_r'] = df_data['text_clean_r'].map(lambda x: remove_url(x, "HTTPURL"))
    df_data['text_clean_r'] = df_data['text_clean_r'].map(lambda x: remove_mentions(x, '@MENTION'))
    df_data['text_clean'] = df_data['text'].map(lambda x: remove_hashtags(x, ''))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_url(x, ""))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_mentions(x, ''))

    print('tagging')

    df_data['pos_tag'] = df_data['text_clean'].map(lambda x: tag(perceptron_tagger, x, sent_tokenizer))
    df_data['no_punctuation'] = df_data['text_clean'].map(lambda x: remove_punctuation(x))
    df_data['no_stopwords'] = df_data['no_punctuation'].map(lambda x: remove_stopwords(x, lang))
    #print('lemmatization')
    #df_data['lemmas'] = df_data['text_clean'].map(lambda x: lemmatize(x))
    df_data['text_clean'] = df_data['text_clean_r']
    df_data = df_data.drop('text_clean_r', 1)
    if not test:
        df_data.to_csv('csv_files/PAN_data_' + lang + '_tagged.csv', sep='\t', encoding='utf-8', index=False)
        print("written to csv")
    print("--- Preprocessing ---", round(((time.time() - start_time) / 60), 2))
    return df_data


def convertToUnicode(df_data):
    # convert to unicode
    df_data['id'] = df_data['id'].map(lambda x: str(x))
    df_data['text'] = df_data['text'].map(lambda x: str(x))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: str(x))
    df_data['variety'] = df_data['variety'].map(lambda x: str(x))
    df_data['gender'] = df_data['gender'].map(lambda x: str(x))
    df_data['pos_tag'] = df_data['pos_tag'].map(lambda x: str(x))
    df_data['no_stopwords'] = df_data['no_stopwords'].map(lambda x: str(x))
    df_data['no_punctuation'] = df_data['no_punctuation'].map(lambda x: str(x))
    #df_data['lemmas'] = df_data['lemmas'].map(lambda x: str(x))
    return df_data



def createFeatures(df_data, sent_tokenizer, lang):
    emoji_dict = get_emojis('word_lists/Emoji_Sentiment_Data_v1.0.csv')
    emoji_list = emoji_dict.keys()
    #model = create_word2vec_model(df_data, sent_tokenizer, lang)


    # read wordlists
    '''emoticons = ['<3', ':D', ':)', ':(', ':>)', ':-)', ':]', '=)', '-(' ':[', '=(', ';)', ';-)', ':-P', ':P', ':-p',
                 ':p', '=P', ':-O', ':O', ':-o', ':o']
    amplifiers = read_wordList('word_lists/amplifier.txt')
    aux_verbs = read_wordList('word_lists/aux_verb.txt')
    cognition_verbs = read_wordList('word_lists/cognition_verb.txt')
    communication_verbs = read_wordList('word_lists/communication_verb.txt')
    modal_verbs = read_wordList('word_lists/modal_verb.txt')
    negations = read_wordList('word_lists/negation.txt')
    fp_pronouns = read_wordList('word_lists/firstperson_pronoun.txt')
    functions = read_wordList('word_lists/function.txt')
    hedge_words = read_wordList('word_lists/hedge_word.txt')
    social_words = read_wordList('word_lists/social.txt')
    swear_words = read_wordList('word_lists/swear_word.txt')
    wh_words = read_wordList('word_lists/wh_word.txt')

    # bieberList
    ability = read_wordList('Biber_simple/Ability_biber.txt')
    attitude = read_wordList('Biber_simple/AttituteEmotion_biber.txt')
    causation = read_wordList('Biber_simple/CausationModalityEffort_biber.txt')
    certainty = read_wordList('Biber_simple/Certainty_biber.txt')
    cognition = read_wordList('Biber_simple/Cognition_biber.txt')
    communication = read_wordList('Biber_simple/Communication_biber.txt')
    desiredecision = read_wordList('Biber_simple/DesireDecision_biber.txt')
    easedifficulty = read_wordList('Biber_simple/EaseDifficulty_biber.txt')
    evaluation = read_wordList('Biber_simple/Evaluation_biber.txt')
    likelihood = read_wordList('Biber_simple/Likelihood_biber.txt')
    necessity = read_wordList('Biber_simple/ModalNecessity_biber.txt')
    possibility = read_wordList('Biber_simple/ModalPossiblity_biber.txt')
    prediction = read_wordList('Biber_simple/ModalPrediction.txt')
    nouns = read_wordList('Biber_simple/Nouns_various.txt')
    premodadv = read_wordList('Biber_simple/PremodAdv_biber.txt')
    style = read_wordList('Biber_simple/Stlye_biber.txt')'''


    df_data['affixes'] = df_data['text_clean'].map(lambda x: get_affix(x))
    df_data['affix_punct'] = df_data['text_clean'].map(lambda x: affix_punct(x))
    #df_data['prefixes'] = df_data['text_clean'].map(lambda x: get_prefix(x))

    # word and POS tag lists
    '''df_data['amplifiers'] = df_data['text_clean'].map(lambda x: countWords(amplifiers,x))
    df_data['aux_verbs'] = df_data['text_clean'].map(lambda x: countWords(aux_verbs, x))
    df_data['cognition_verbs'] = df_data['text_clean'].map(lambda x: countWords(cognition_verbs, x))
    df_data['communication_verbs'] = df_data['text_clean'].map(lambda x: countWords(communication_verbs, x))
    df_data['modal_verbs'] = df_data['text_clean'].map(lambda x: countWords(modal_verbs, x))
    df_data['negations'] = df_data['text_clean'].map(lambda x: countWords(negations, x))
    df_data['fp_pronouns'] = df_data['text_clean'].map(lambda x: countWords(fp_pronouns, x))
    df_data['functions'] = df_data['text_clean'].map(lambda x: countWords(functions, x))
    df_data['wh_words'] = df_data['text_clean'].map(lambda x: countWords(wh_words, x))

    df_data['number_of_nouns'] = df_data['pos_tag'].map(lambda x: count_pos(x, ['NN', 'NNP', 'NNS', 'NNPS']))
    df_data['number_of_pronouns'] = df_data['pos_tag'].map(lambda x: count_pos(x, ['PRP', 'PRP$']))

    df_data['hedge_words'] = df_data['text_clean'].map(lambda x: countWords(hedge_words, x))
    df_data['social_words'] = df_data['text_clean'].map(lambda x: countWords(social_words, x))
    df_data['swear_words'] = df_data['text_clean'].map(lambda x: countWords(swear_words, x))
    df_data['number_of_errors'] = df_data['text_clean'].map(lambda x: count_error(x,chkr))

    #biber
    df_data['necessity'] = df_data['lemmas'].map(lambda x: countWords(necessity, x))
    df_data['ability'] = df_data['lemmas'].map(lambda x: countWords(ability, x))
    df_data['attitude'] = df_data['lemmas'].map(lambda x: countWords(attitude, x))
    df_data['causation'] = df_data['lemmas'].map(lambda x: countWords(causation, x))
    df_data['certainty'] = df_data['lemmas'].map(lambda x: countWords(certainty, x))
    df_data['cognition'] = df_data['lemmas'].map(lambda x: countWords(cognition, x))
    df_data['communication'] = df_data['lemmas'].map(lambda x: countWords(communication, x))
    df_data['desiredecision'] = df_data['lemmas'].map(lambda x: countWords(desiredecision, x))
    df_data['easedifficulty'] = df_data['lemmas'].map(lambda x: countWords(easedifficulty, x))
    df_data['evaluation'] = df_data['lemmas'].map(lambda x: countWords(evaluation, x))
    df_data['likelihood'] = df_data['lemmas'].map(lambda x: countWords(likelihood, x))
    df_data['possibility'] = df_data['lemmas'].map(lambda x: countWords(possibility, x))
    df_data['prediction'] = df_data['lemmas'].map(lambda x: countWords(prediction, x))
    df_data['nouns'] = df_data['lemmas'].map(lambda x: countWords(nouns, x))
    df_data['premodadv'] = df_data['lemmas'].map(lambda x: countWords(premodadv, x))
    df_data['style'] = df_data['text_clean'].map(lambda x: countWords(style, x))'''

    #non language specific features

    #df_data['number_of_commas'] = df_data['text_clean'].map(lambda x: countWords([','], x))
    #df_data['number_of_question_marks'] = df_data['text_clean'].map(lambda x: countWords(['?'], x))
    #df_data['number_of_dots'] = df_data['text_clean'].map(lambda x: countWords(['.'], x))
    #df_data['number_of_exclamation_marks'] = df_data['text_clean'].map(lambda x: countWords(['!'], x))
    #df_data['number_of_punct'] = df_data['text_clean'].map(lambda x: countWords(['!', ',', '?', '.'], x))
    #df_data['unique_words'] = df_data['text_clean'].map(lambda x: len(set(x.split()))/len(x.split()))
    #df_data['text_length'] = df_data['text_clean'].map(lambda x: len(x.split()))
    df_data['number_of_emojis'] = df_data['text_clean'].map(lambda x: count_patterns(x, emoji_list))
    df_data['sentiment'] = df_data['text_clean'].map(lambda x: get_sentiment(x, emoji_dict))
    df_data['number_of_character_floods'] = df_data['no_punctuation'].map(lambda x: countCharacterFlooding(x))
    #df_data['w2v'] = df_data['text_clean'].map(lambda x: makeDocVecFeatures(x, model, sent_tokenizer))

    #df_data['text_clean'] = df_data['text_clean'].map(lambda x: compress(x))
    #df_data['no_stopwords'] = df_data['no_stopwords'].map(lambda x: compress(x))
    #df_data['no_punctuation'] = df_data['no_punctuation'].map(lambda x: compress(x))
    #df_data['count_digits'] = df_data['text_clean'].map(lambda x: len(re.findall(r"([\d.]*\d+)", x)))
    #df_data['count_currencies'] = df_data['text_clean'].map(lambda x: len(re.findall(r"[£$€]", x)))
    #df_data['number_of_emoticons'] = df_data['text_clean'].map(lambda x: count_patterns(x, emoticons))
    #df_data['number_of_mentions'] = df_data['text_clean'].map(lambda x: count_patterns(x, ['@MENTION']))
    #df_data['number_of_urls'] = df_data['text_clean'].map(lambda x: count_patterns(x, ['HTTPURL']))
    #df_data['number_of_hashtags'] = df_data['text_clean'].map(lambda x: count_patterns(x, ['#HASHTAG']))
    #df_data['number_of_mentions_hashtags'] = df_data['text_clean'].map(lambda x: count_patterns(x, ['@MENTION', '#HASHTAG']))

    #network features
    #df_network = readNetworkFeatures()
    #df_data = pd.concat([df_data, df_network], axis=1)
    return df_data

def get_stats(df_data, lang):

    #get some stats
    print('all authors: ', df_data.shape[0])
    df_male = df_data[df_data['gender'].isin(['male' ,'M', 'm'])]
    df_female = df_data[df_data['gender'].isin(['female' ,'F', 'f'])]

    print('num. males: ', df_male.shape[0])
    print('num. females: ', df_female.shape[0])
    print('majority class. for gender: ', max(df_male.shape[0], df_female.shape[0])/df_data.shape[0])

    if lang == 'pt':
        df_por = df_data[df_data['variety'] == 'portugal']
        df_bra = df_data[df_data['variety'] == 'brazil']
        print('num. aus: ', df_por.shape[0])
        print('num. can: ', df_bra.shape[0])
        print('majority class variety: ', max(df_por.shape[0], df_bra.shape[0]) / df_data.shape[0])

    elif lang == 'ar':
        df_gul = df_data[df_data['variety'] == 'gulf']
        df_lev = df_data[df_data['variety'] == 'levantine']
        df_mag = df_data[df_data['variety'] == 'maghrebi']
        df_egy = df_data[df_data['variety'] == 'egypt']

        print('num. gulf: ', df_gul.shape[0])
        print('num. levantine: ', df_lev.shape[0])
        print('num. maghrebi: ', df_mag.shape[0])
        print('num. egypti: ', df_egy.shape[0])
        print('majority class variety: ',max(df_egy.shape[0], df_gul.shape[0], df_lev.shape[0], df_mag.shape[0]) / df_data.shape[0])

    elif lang == 'en':
        df_aus = df_data[df_data['variety'] == 'australia']
        df_can = df_data[df_data['variety'] == 'canada']
        df_gb = df_data[df_data['variety'] == 'great britain']
        df_ire = df_data[df_data['variety'] == 'ireland']
        df_nz = df_data[df_data['variety'] == 'new zealand']
        df_us = df_data[df_data['variety'] == 'united states']
        print('num. aus: ', df_aus.shape[0])
        print('num. can: ', df_can.shape[0])
        print('num. gb: ', df_gb.shape[0])
        print('num. ire: ', df_ire.shape[0])
        print('num. nz: ', df_nz.shape[0])
        print('num. us: ', df_us.shape[0])
        print('majority class variety: ', max(df_aus.shape[0], df_can.shape[0], df_gb.shape[0], df_ire.shape[0], df_nz.shape[0], df_us.shape[0]) / df_data.shape[0])

    elif lang == 'es':
        df_arg = df_data[df_data['variety'] == 'argentina']
        df_chi = df_data[df_data['variety'] == 'chile']
        df_col = df_data[df_data['variety'] == 'colombia']
        df_mex = df_data[df_data['variety'] == 'mexico']
        df_per = df_data[df_data['variety'] == 'peru']
        df_spa = df_data[df_data['variety'] == 'spain']
        df_ven = df_data[df_data['variety'] == 'venezuela']
        print('num. arg: ', df_arg.shape[0])
        print('num. chi: ', df_chi.shape[0])
        print('num. col: ', df_col.shape[0])
        print('num. mex: ', df_mex.shape[0])
        print('num. per: ', df_per.shape[0])
        print('num. spa: ', df_spa.shape[0])
        print('num. ven: ', df_ven.shape[0])
        print('majority class variety: ', max(df_arg.shape[0], df_chi.shape[0], df_col.shape[0], df_mex.shape[0], df_per.shape[0], df_spa.shape[0], df_ven.shape[0]) / df_data.shape[0])



