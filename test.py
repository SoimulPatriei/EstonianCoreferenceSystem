
"""The testing module of the Estonian Pronominal Coreference System. """

__author__ = "Eduard Barbu"
__license__ = "LGPL"
__version__ = "1.0.0"
__maintainer__ = "Eduard Barbu"
__email__ = "barbu@ut.ee"

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import utilities
import generate_pairs
import coreference_features
import argparse
import sys
import os
import collections
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def add_mentions (pronoun_list,corpus_file_list,tagged_corpus_dir,exlcude_list) :
    tagged_corpus_list=[]
    nlp=utilities.init_estonian_pipeline()
    for i, f_corpus_path in enumerate(corpus_file_list):
        corpus_path, f_name = os.path.split(f_corpus_path)
        f_corpus_tagged_path=os.path.join(tagged_corpus_dir,f_name)
        tagged_corpus_list.append(f_corpus_tagged_path)
        logging.info(f"add_mentions::Tag {f_corpus_path}=>{f_corpus_tagged_path}")
        doc=utilities.compute_mentions(nlp,f_corpus_path,pronoun_list,exlcude_list)
        utilities.serialize_text(doc,f_corpus_tagged_path)
    return tagged_corpus_list

def update_index (dict_index,dict_features)  :
    for key in dict_features :
        dict_index[key]=0

def generate_features (corpus_file_list, dict_catalog,sklearn_test_file,tagged_corpus_dir) :
    """Generate the scikit-learn feature file to be passed to the ML algorithms"""

    dict_index = collections.OrderedDict()
    dict_info = {
        "context": utilities.read_context_file(dict_catalog["sentence_context_file"]),
        "tagset": utilities.read_configuration_file(dict_catalog["tagset_file"], "pos"),
        "cases": utilities.read_configuration_file(dict_catalog["cases_file"], "case"),
        "syntactic_functions": utilities.read_syntactic_file(dict_catalog["syntactic_function_file"],"syntactic_function"),
        "exclude_list" : utilities.read_exclude_words(dict_catalog["mention_info"])
    }

    tagged_corpus_list=add_mentions(dict_info["context"].keys(), corpus_file_list, tagged_corpus_dir,dict_info["exclude_list"])

    for i,f_corpus_path in enumerate(tagged_corpus_list):
        logging.info(f"generate_features::{f_corpus_path}")
        sentences_list = utilities.deserialize_file(f_corpus_path)
        dict_features = generate_pairs.pronominal_coreference_candidate_pairs(dict_info,sentences_list, f_corpus_path)
        update_index(dict_index, dict_features)
        if i==0 :
            dict_features_names=coreference_features.get_feature_names()
            utilities.generate_scikit_learn_antet(dict_features_names, sklearn_test_file)
        utilities.append_scikit_learn_file(dict_features, sklearn_test_file)
    return dict_index

def classify (sklearn_test_file,f_feature_names,dict_index,f_model,f_output) :
    fo = open(f_output, mode='w', encoding='utf-8')

    dict_feature_type = utilities.get_feature_type(f_feature_names)
    categorical_features = [feature for feature in dict_feature_type if dict_feature_type[feature] == 'categorical']

    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto',handle_unknown='ignore'), categorical_features)],remainder = 'passthrough')
    pipeline = Pipeline(steps=[('t', ct), ('m', XGBClassifier())])

    X_train, y_train, features = utilities.getXy(f_model)
    pipeline.fit(X_train, y_train)

    X_test = utilities.get_X(sklearn_test_file, features)
    y_pred = pipeline.predict(X_test)

    for i, coreference_pair in enumerate(dict_index.keys()) :
        if y_pred[i] :
            fo.write(coreference_pair+"\n")
    fo.close()

def create_directory (dir_path) :

    if os.path.exists(dir_path) :
            logging.info(f"The directory {dir_path } exists")
            return
    try:
        os.mkdir(dir_path)
    except OSError:
        logging.error(f"Creation of the directory {dir_path} failed" )
        sys.exit()
    else:
        logging.info(f"Successfully created the directory {dir_path }")


def test () :
    """Test your classifier on the test corpus"""

    parser = argparse.ArgumentParser(
        description='Get the pronominal coreference pairs using the trained model')
    parser.add_argument("--catalog", type=str, help="The relative path to your language catalog of resources")
    parser.add_argument("--corpus_dir", type=str, help="The relative path to your language test corpus")
    parser.add_argument("--corpus_tagged_dir", type=str, help="The relative path to the language corpus that will be tagged")
    parser.add_argument("--sklearn_file_training", type=str,help="The relative path to the stored trained model")
    parser.add_argument("--sklearn_file_test", type=str,help="The relative path to the sklearn file computed for the test corpus")
    parser.add_argument("--feature_names_file", type=str, help="The file containing the names of the computed features")
    parser.add_argument("--output_file", type=str, help="The file containing the computed coreference pairs")

    args = parser.parse_args()

    if not (args.catalog and args.corpus_dir and args.corpus_tagged_dir and args.sklearn_file_training and
            args.sklearn_file_test and args.feature_names_file  and args.output_file) :
        logging.error("Wrong arguments!")
        sys.exit()

    create_directory(args.corpus_tagged_dir)

    dict_catalog = utilities.read_resource_catalog(args.catalog)
    logging.info(f"test::Read Resource Catalog from=>{args.catalog}")

    corpus_file_list = utilities.read_corpus(args.corpus_dir,"*.txt")
    logging.info(f"test::Read the test corpus file names from=>{args.corpus_dir}")

    coreference_features.get_mention_global_scores (dict_catalog["global_mention_scores"])
    logging.info(f"""test::Read the global mention scores from=>{dict_catalog["global_mention_scores"]}""")

    coreference_features.get_eleri_abstractness (dict_catalog["eleri_abstractness"])
    logging.info(f"""test::Read Eleri Aedmaa abstractness scores from=> {dict_catalog["eleri_abstractness"]}""")

    coreference_features.init_embedding_models(dict_catalog["embeddings_file"], logging)
    logging.info(f"""test::Inited the embedding models from=> {dict_catalog["embeddings_file"]}""")

    logging.info(f"test::Obtain the coreference pairs from the corpus=>{args.corpus_dir}\n")
    dict_index=generate_features(corpus_file_list, dict_catalog, args.sklearn_file_test, args.corpus_tagged_dir)

    logging.info (f"test::classification...")
    classify(args.sklearn_file_test, args.feature_names_file, dict_index,args.sklearn_file_training,  args.output_file)
    logging.info(f"test::result in {args.output_file}")

def main():
    test()


if __name__ == '__main__':
    main()