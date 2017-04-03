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

import resource
rsrc = resource.RLIMIT_AS
soft, hard = resource.getrlimit(rsrc)
#resource.setrlimit(rsrc, (13500000000, hard)) #limit allowed Python memory usage to 13GB

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
def readPANcorpus(path, lang):
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
            if document.text:
                cntTweet += 1
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
    print(cntRT)
    print('Tweets: ', cntTweet)

    #write to csv file
    with open('PAN_data_' + lang + '.csv', 'w') as fp:
        a = csv.writer(fp, delimiter='\t')
        a.writerows(data)
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


def remove_stopwords(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)


def remove_mentions(text):
    return re.sub(r'(?:@[\w_]+)', '', text)


def remove_hashtags(text):
    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', text)


def remove_url(text):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, "", text)


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

#count emoticons and return frequency
def count_emoticons(text, list):
    cnt=0
    length = len(text)
    for emoticon in list:
        cnt += text.count(emoticon)
    if length == 0:
        return 0
    return cnt/length

#count specific pos tags and return frequency
def count_pos(pos_sequence, pos_list):
    cnt = 0
    for pos_tag in pos_sequence.split():
        for pos in pos_list:
            if pos_tag == pos:
                cnt += 1
    return cnt/len(pos_sequence.split())


#create noun pronoun ratio feature
def noun_pronoun_ratio(text):
    noun_words = ['NN', 'NNP', 'NNS', 'NNPS', 'IN', 'DT', 'CD']
    pronoun_words = ['PRP', 'PRP$']
    pronoun_words.extend(list(aux_verbs))
    pronoun_words.extend(list(hedge_words))
    cnt_noun = 0
    cnt_pronoun = 0
    for noun in noun_words:
        cnt_noun += text.count(noun)
    for pronoun in pronoun_words:
        cnt_pronoun += text.count(pronoun)
    if cnt_pronoun == 0:
        return 0
    return cnt_noun/cnt_pronoun


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
    text = remove_hashtags(text)
    text = remove_mentions(text)
    text = remove_url(text)
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
    return " ".join([word[-4:] if len(word) >=4 else word for word in text.split()])


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
        d_col_drops=['text', 'pos_tag', 'no_punctuation', 'no_stopwords', 'text_clean', 'lemmas', 'affixes']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
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


if __name__ == '__main__':

    #run from command line
    argparser = argparse.ArgumentParser(description='Author Profiling Evaluation')
    argparser.add_argument('-l', '--language', dest='language', type=str, default='en',
                           help='Set language')

    argparser.add_argument('-t', '--task', dest='task', type=str, default='gender',
                           help='Set task')

    argparser.add_argument('-m', '--create_model', dest='create_model', type=str, default='True',
                           help='Choose to create model or not')

    argparser.add_argument('-o', '--output', dest='output', type=str, default='./results',
                           help='Choose output directory')

    argparser.add_argument('-c', '--input', dest='input', type=str, default='pan17-author-profiling-training-dataset-2017-03-10',
                           help='Choose input trainset')

    args = argparser.parse_args()

    #uncomment this to create new classification model, otherwise use a pickled model and test it on blogs
    create_model = args.create_model
    create_model = True if create_model == 'True' else False
    lang = args.language
    task = args.task
    output = args.output
    input = args.input

    print(create_model, lang, task)
    #train tagger
    if lang == 'en':
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        perceptron_tagger = PerceptronTagger()
    else:
        perceptron_tagger = PerceptronTagger(load=False)
        if lang == 'es':
            sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
            perceptron_tagger.train(list(cess.tagged_sents()))
        elif lang == 'pt':
            sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
            tsents = floresta.tagged_sents()
            tsents = [[(w.lower(), simplify_tag(t)) for (w, t) in sent] for sent in tsents if sent]
            perceptron_tagger.train(tsents)
        else:
            sent_tokenizer = None

    #read wordlists
    emoticons = ['<3', ':D', ':)', ':(', ':>)', ':-)', ':]', '=)', '-(' ':[', '=(', ';)', ';-)', ':-P', ':P', ':-p', ':p', '=P', ':-O', ':O', ':-o', ':o']
    #emoticons = read_wordList('emoticon.txt')
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

    #bieberList
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
    style = read_wordList('Biber_simple/Stlye_biber.txt')


    # uncomment this section if you want to read train sets from original file
    readPANcorpus(input, lang)
    if not create_model:
        df_data = pd.read_csv('PAN_data_' + lang + '.csv', encoding="utf-8", delimiter="\t")[:100]
    else:
        df_data = pd.read_csv('PAN_data_' + lang + '.csv', encoding="utf-8", delimiter="\t")
    print(df_data.shape)

    #preprocess and tag data and write it to csv for later use
    #df_data['text'] = df_data['text'].values.astype('U')
    print('clean text')
    #df_data['text'] = df_data['text'].map(lambda x: beautify(x))
    df_data['text_clean'] = df_data['text'].map(lambda x: remove_hashtags(x))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_url(x))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_mentions(x))
    print('tagging')
    df_data['pos_tag'] = df_data['text_clean'].map(lambda x: tag(perceptron_tagger, x, sent_tokenizer))
    df_data['no_punctuation'] = df_data['text_clean'].map(lambda x: remove_punctuation(x))
    df_data['no_stopwords'] = df_data['no_punctuation'].map(lambda x: remove_stopwords(x))
    print('lemmatization')
    df_data['lemmas'] = df_data['text_clean'].map(lambda x: lemmatize(x))

    df_data.to_csv('PAN_data_' + lang + '_tagged.csv', sep='\t', encoding='utf-8', index=False)
    print("written to csv")

    #uncomment this to read  data from csv
    data_iterator = pd.read_csv('PAN_data_' + lang + '_tagged.csv', encoding="utf-8", delimiter="\t", chunksize=1000)
    df_data = pd.DataFrame()
    for sub_data in data_iterator:
        df_data = pd.concat([df_data, sub_data], axis=0)
        gc.collect()
    print("read csv")
    print(df_data.columns.tolist())
    print(df_data.shape)

    #shuffle the corpus and optionaly choose the chunk you want to use if you don't want to use the whole thing - will be much faster
    #df_data = df_data.reindex(np.random.permutation(df_data.index))


    # convert to unicode
    df_data['id'] = df_data['id'].map(lambda x: str(x))
    df_data['text'] = df_data['text'].map(lambda x: str(x))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: str(x))
    df_data['variety'] = df_data['variety'].map(lambda x: str(x))
    df_data['gender'] = df_data['gender'].map(lambda x: str(x))
    df_data['pos_tag'] = df_data['pos_tag'].map(lambda x: str(x))
    df_data['no_stopwords'] = df_data['no_stopwords'].map(lambda x: str(x))
    df_data['no_punctuation'] = df_data['no_punctuation'].map(lambda x: str(x))
    df_data['lemmas'] = df_data['lemmas'].map(lambda x: str(x))

    df_data['affixes'] = df_data['text_clean'].map(lambda x: get_affix(x))

    # create numeric features
    df_data['amplifiers'] = df_data['text_clean'].map(lambda x: countWords(amplifiers,x))
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
    df_data['number_of_dots'] = df_data['text_clean'].map(lambda x: countWords(['.'], x))
    df_data['number_of_exclamation_marks'] = df_data['text_clean'].map(lambda x: countWords(['!'], x))

    df_data['hedge_words'] = df_data['text_clean'].map(lambda x: countWords(hedge_words, x))
    df_data['social_words'] = df_data['text_clean'].map(lambda x: countWords(social_words, x))
    df_data['swear_words'] = df_data['text_clean'].map(lambda x: countWords(swear_words, x))
    df_data['number_of_emoticons'] = df_data['text_clean'].map(lambda x: count_emoticons(x, emoticons))
    df_data['number_of_commas'] = df_data['text_clean'].map(lambda x: countWords([','], x))
    df_data['number_of_question_marks'] = df_data['text_clean'].map(lambda x: countWords(['?'], x))
    df_data['number_of_errors'] = df_data['text_clean'].map(lambda x: count_error(x,chkr))
    df_data['noun_pronoun_ratio'] = (df_data['text_clean'] + " " + df_data['pos_tag']).map(lambda x: noun_pronoun_ratio(x))

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
    df_data['style'] = df_data['text_clean'].map(lambda x: countWords(style, x))

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


    print("--- Preprocessing ---", round(((time.time() - start_time)/60),2))

    if create_model:
        #numeric feature evaluation
        X_eval = df_data.drop(['gender','variety','text', 'pos_tag', 'no_punctuation', 'no_stopwords', 'text_clean', 'lemmas', 'affixes', 'id'], axis=1)
        column_names = X_eval.columns.values
        minmax_scale = preprocessing.MinMaxScaler().fit(X_eval)
        X_eval=minmax_scale.transform(X_eval)
        y_eval = df_data['variety'].values
        chi2score = chi2(X_eval, y_eval)
        chi2score = zip(chi2score[0], chi2score[1])
        wscores = zip(column_names, chi2score)
        result_list = sorted(wscores, reverse=True, key=lambda x: x[1][0])
        print("Numeric features arranged by chi2 value - the other number is p-value")
        print(result_list)

        #choose to predict either gender or age
        y = df_data[task].values
        X = df_data.drop(['gender', 'variety', 'id'], axis=1)


        #other feature evaluation
        mnb = MultinomialNB()
        tfidf = TfidfVectorizer(ngram_range=(3,3), lowercase=False)
        trainset= tfidf.fit_transform(df_data['pos_tag'].values)
        mnb.fit(trainset, y)
        most_informative_feature_for_class(tfidf, mnb)


        #build classification model
        svm = SVC(decision_function_shape ='ovr', C=1.0, kernel="linear", probability=True)
        lsvm = LinearSVC(penalty='l2', multi_class='ovr', fit_intercept=False, C=1.0)
        lr = LogisticRegression(C=1e2, multi_class='ovr', solver='liblinear', fit_intercept=False, random_state=123)
        rfc = RandomForestClassifier(random_state=2016, n_estimators=200, max_depth=15)
        eclf = VotingClassifier(estimators=[('lr', lr), ('svm', lsvm)], voting="hard")
        bclf = BaggingClassifier(base_estimator=svm, random_state=2016, max_samples=0.7, max_features=0.7, n_estimators=100)
        xgb = xgb.XGBClassifier(max_depth=5, subsample=0.8, n_estimators=1000, min_child_weight=1, colsample_bytree=0.8, learning_rate=1, nthread=8)
        baseline = DummyClassifier(strategy='most_frequent')
        tfidf_unigram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=10, max_df=0.8)
        tfidf_bigram = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=False, min_df=20, max_df=0.5)
        tfidf_topics = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=False, min_df=1, max_df=0.5)
        tfidf_pos = TfidfVectorizer(ngram_range=(2,2), sublinear_tf=True, min_df=0.1, max_df=0.6, lowercase=False)
        character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), lowercase=False, min_df=4, max_df=0.8)
        tfidf_affixes = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.8)
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        tsvd = TruncatedSVD(random_state=2016, n_components=200, n_iter=5)

        features = [#('cst', digit_col()),
                    ('unigram', pipeline.Pipeline([('s1', text_col(key='no_stopwords')), ('tfidf_unigram', tfidf_unigram)])),
                    ('bigram', pipeline.Pipeline([('s2', text_col(key='no_punctuation')), ('tfidf_bigram', tfidf_bigram)])),
                    #('topics', pipeline.Pipeline([('s3', text_col(key='no_stopwords')), ('tfidf_topics', tfidf_topics), ('tsvd', tsvd)])),
                    ('tag', pipeline.Pipeline([('s4', text_col(key='pos_tag')), ('tfidf_pos', tfidf_pos)])),
                    ('character', pipeline.Pipeline([('s5', text_col(key='text_clean')),('character_vectorizer', character_vectorizer), ('tfidf_character', tfidf_transformer)])),
                    ('suffixes', pipeline.Pipeline([('s5', text_col(key='affixes')),('tfidf_affixes', tfidf_affixes)])),
                    ]
        weights = {#'cst': 0.2,
                   'unigram': 0.8,
                   'bigram': 0.1,
                   #'topics': 0.1,
                   'tag': 0.2,
                   'character': 0.9,
                   'suffixes': 0.4
        }

        if lang == 'ar':
            features = features[0:2] + features[3:]
            del weights['tag']

        clf = pipeline.Pipeline([
                ('union', FeatureUnion(
                        transformer_list = features,
                        transformer_weights = weights,
                        n_jobs = 1
                        )),
                ('scale', Normalizer()),
                ('svm', lr)])
        kfold = model_selection.KFold(n_splits=10, random_state=2016)
        results = model_selection.cross_val_score(clf, X , y, cv=kfold, verbose=20)
        print("CV score:")
        print(results.mean())

        clf.fit(X, y)
        joblib.dump(clf, 'svm_clf_' + lang + '_' + task + '.pkl')
        print("--- Model creation in minutes ---", round(((time.time() - start_time) / 60), 2))
        print("--- Training & Testing in minutes ---", round(((time.time() - start_time) / 60), 2))

    else:
        id = df_data['id']
        X = df_data.drop(['gender', 'variety', 'id'], axis=1)
        clf = joblib.load('svm_clf_' + lang + '_gender.pkl')
        y_pred_gender = clf.predict(X)
        clf = joblib.load('svm_clf_' + lang + '_variety.pkl')
        y_pred_variety = clf.predict(X)
        print("Accuracy on test set:")
        df_results = pd.DataFrame({"id": id, "gender": y_pred_gender, "variety": y_pred_variety})
        df_results.to_csv('results_' + lang + '.csv', index=False, header=True)
        for index, row in df_results.iterrows():
            generate_output(output, row['id'], lang, row['variety'], row['gender'])


