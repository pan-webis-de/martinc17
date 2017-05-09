from main import *
import os

if __name__ == '__main__':
    # run from command line
    # e.g. python3 main.py --input './pan17-author-profiling-training-dataset-2017-03-10' --output results --language en
    argparser = argparse.ArgumentParser(description='Author Profiling Evaluation')
    argparser.add_argument('-l', '--language', dest='language', type=str, default='en',
                           help='Set language')

    argparser.add_argument('-t', '--task', dest='task', type=str, default='variety',
                           help='Set task')

    argparser.add_argument('-c', '--input', dest='input', type=str,
                           default='pan17-author-profiling-training-dataset-2017-03-10',
                           help='Choose input trainset')
    args = argparser.parse_args()

    lang = args.language
    task = args.task
    input = args.input

    print(lang, task, input)

    # train tagger
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

    # uncomment this section if you want to read train sets from original file
    #df_data = readPANcorpus(input, lang)
    #df_data = pd.read_csv('csv_files/PAN_data_' + lang + '.csv', encoding="utf-8", delimiter="\t")
    #print("Data shape: ", df_data.shape)
    #df_data = preprocess(df_data, lang, perceptron_tagger, sent_tokenizer)

    # uncomment this to read  data from csv
    data_iterator = pd.read_csv('csv_files/PAN_data_' + lang + '_tagged.csv', encoding="utf-8", delimiter="\t", chunksize=1000)
    df_data = pd.DataFrame()
    for sub_data in data_iterator:
        df_data = pd.concat([df_data, sub_data], axis=0)
        gc.collect()
    print("read csv")
    print(df_data.columns.tolist())
    print("Data shape after preprocessing:", df_data.shape)

    # shuffle the corpus and optionaly choose the chunk you want to use if you don't want to use the whole thing - will be much faster
    #df_data = df_data.reindex(np.random.seed(42))
    #df_data = df_data[:1000]

    df_data = convertToUnicode(df_data)
    df_data = createFeatures(df_data, sent_tokenizer, lang)
    #get_stats(df_data, lang)

    # numeric feature evaluation
    X_eval = df_data.drop(
        ['gender', 'variety', 'text', 'pos_tag', 'no_punctuation', 'no_stopwords', 'text_clean', 'lemmas', 'affixes', 'mid_punct', 'id'], axis=1)
    column_names = X_eval.columns.values
    minmax_scale = preprocessing.MinMaxScaler().fit(X_eval)
    X_eval = minmax_scale.transform(X_eval)
    y_eval = df_data['variety'].values
    chi2score = chi2(X_eval, y_eval)
    chi2score = zip(chi2score[0], chi2score[1])
    wscores = zip(column_names, chi2score)
    result_list = sorted(wscores, reverse=True, key=lambda x: x[1][0])
    print("Numeric features arranged by chi2 value - the other number is p-value")
    print(result_list)

    y = df_data[task].values
    X = df_data.drop(['gender', 'variety', 'id'], axis=1)

    #zusammen nehmen gender und variety
    #df_data['variety_gender'] = df_data['gender'] + "_" + df_data['variety']
    #y = df_data['variety_gender'].values
    #X = df_data.drop(['gender', 'variety', 'id', 'variety_gender'], axis=1)

    # other feature evaluation
    mnb = MultinomialNB()
    tfidf = TfidfVectorizer(ngram_range=(3, 3), lowercase=False)
    trainset = tfidf.fit_transform(df_data['pos_tag'].values)
    mnb.fit(trainset, y)
    #most_informative_feature_for_class(tfidf, mnb)

    # build classification model
    svm = SVC(decision_function_shape='ovr', C=1.0, kernel="linear", probability=True)
    lsvm = LinearSVC(penalty='l2', multi_class='ovr', fit_intercept=False, C=1.0)
    lr = LogisticRegression(C=1e2, multi_class='ovr', solver='liblinear', fit_intercept=False, random_state=123)
    rfc = RandomForestClassifier(random_state=2016, n_estimators=200, max_depth=15)
    eclf = VotingClassifier(estimators=[('lr', lr), ('svm', lsvm)], voting="hard")
    bclf = BaggingClassifier(base_estimator=svm, random_state=2016, max_samples=0.7, max_features=0.7, n_estimators=100)
    xgb = xgb.XGBClassifier(max_depth=5, subsample=0.8, n_estimators=1000, min_child_weight=1, colsample_bytree=0.8,
                            learning_rate=1, nthread=8)
    baseline = DummyClassifier(strategy='most_frequent')
    tfidf_unigram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=10, max_df=0.8)
    tfidf_bigram = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=False, min_df=20, max_df=0.5)
    tfidf_topics = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=False, min_df=1, max_df=0.5)
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=True, min_df=0.1, max_df=0.6, lowercase=False)
    character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), lowercase=False, min_df=4,
                                           max_df=0.8)
    tfidf_ngram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.8)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    tsvd = TruncatedSVD(random_state=2016, n_components=200, n_iter=5)

    features = [('cst', digit_col()),
        ('unigram', pipeline.Pipeline([('s1', text_col(key='no_stopwords')), ('tfidf_unigram', tfidf_unigram)])),
        ('bigram', pipeline.Pipeline([('s2', text_col(key='no_punctuation')), ('tfidf_bigram', tfidf_bigram)])),
        #('topics', pipeline.Pipeline([('s3', text_col(key='no_stopwords')), ('tfidf_topics', tfidf_topics), ('tsvd', tsvd)])),
        ('tag', pipeline.Pipeline([('s4', text_col(key='pos_tag')), ('tfidf_pos', tfidf_pos)])),
        ('character', pipeline.Pipeline([('s5', text_col(key='text_clean')), ('character_vectorizer', character_vectorizer),
            ('tfidf_character', tfidf_transformer)])),
        ('affixes', pipeline.Pipeline([('s5', text_col(key='affixes')), ('tfidf_ngram', tfidf_ngram)])),
        ('mid_punct', pipeline.Pipeline([('s5', text_col(key='mid_punct')), ('tfidf_ngram', tfidf_ngram)])),
        #('w2v', pipeline.Pipeline([('s5', w2v_col(key='w2v'))])),
    ]
    weights = {'cst': 0.3,
        'unigram': 0.8,
        'bigram': 0.1,
        #'topics': 0.1,
        'tag': 0.2,
        'character': 0.8, #0.8274
        'affixes': 0.3,
        'mid_punct': 0.2,
        #'w2v':0.1,
    }

    if lang == 'ar':
        features = features[0:3] + features[4:]
        del weights['tag']

    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=features,
            transformer_weights=weights,
            n_jobs=1
        )),
        ('scale', Normalizer()),
        ('svm', lr)])
    kfold = model_selection.KFold(n_splits=10, random_state=2016)
    results = model_selection.cross_val_score(clf, X, y, cv=kfold, verbose=20)
    print("CV score:")
    print(results.mean())

    clf.fit(X, y)
    joblib.dump(clf, 'models/svm_clf_' + lang + '_' + task + '.pkl')
    print("--- Model creation in minutes ---", round(((time.time() - start_time) / 60), 2))
    print("--- Training & Testing in minutes ---", round(((time.time() - start_time) / 60), 2))
