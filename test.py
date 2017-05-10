from main import *
import os
import shutil

if __name__ == '__main__':
    #run from command line
    #e.g. python3 main.py  --input './pan17-author-profiling-training-dataset-2017-03-10' --output results
    argparser = argparse.ArgumentParser(description='Author Profiling Evaluation')
    argparser.add_argument('-o', '--output', dest='output', type=str, default='./results',
                           help='Choose output directory')

    argparser.add_argument('-c', '--input', dest='input', type=str, default='pan17-author-profiling-training-dataset-2017-03-10',
                           help='Choose input trainset')
    args = argparser.parse_args()

    args = argparser.parse_args()

    output = args.output
    input = args.input

    #train taggers
    for lang in ['pt', 'es', 'en', 'ar']:
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

        df_data = readPANcorpus(input, lang, test=True)
        print("Language: ", lang)
        df_data = preprocess(df_data, lang, perceptron_tagger, sent_tokenizer, test=True)
        df_data = convertToUnicode(df_data)
        df_data = createFeatures(df_data, sent_tokenizer, lang)
        print("Data shape: ", df_data.shape)
        #test model
        id = df_data['id']
        X = df_data.drop(['gender', 'variety', 'id'], axis=1)
        clf = joblib.load('models/lr_clf_' + lang + '_gender.pkl')
        y_pred_gender = clf.predict(X)
        clf = joblib.load('models/lr_clf_' + lang + '_variety.pkl')
        y_pred_variety = clf.predict(X)
        df_results = pd.DataFrame({"id": id, "gender": y_pred_gender, "variety": y_pred_variety})
        #df_results.to_csv('csv_files/results_' + lang + '.csv', index=False, header=True)
        if os.path.exists(output + '/' + lang):
            shutil.rmtree(output + '/' + lang)
        os.makedirs(output + '/' + lang)
        for index, row in df_results.iterrows():
            generate_output(output + '/' + lang, row['id'], lang, row['variety'], row['gender'])