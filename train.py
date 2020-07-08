"""The training module of the Estonian Pronominal Coreference System. """

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

def generate_feature_names_file (dict_features_names,feature_names_file) :

    fo = open(feature_names_file, mode='w', encoding='utf-8')
    for feature in dict_features_names:
        fo.write(feature +"\t" + dict_features_names[feature] + "\n")
    fo.close()

def add_category (dict_features,sentences_list,f_corpus_path) :
    """Add the category to the generated features"""

    signature_list=generate_pairs.get_annotated_pairs(sentences_list,f_corpus_path)
    for key in dict_features :
        if key in signature_list :
            dict_features[key]["category"]=1
            signature_list.remove(key)
        else :
            dict_features[key]["category"] = 0

def generate_features (corpus_file_list, dict_catalog, sklearn_file,feature_names_file) :

    """Generate the scikit-learn feature file to be passed to the ML algorithms"""

    dict_info={
               "context": utilities.read_context_file(dict_catalog["sentence_context_file"]),
               "tagset" : utilities.read_configuration_file(dict_catalog["tagset_file"], "pos"),
               "cases" :  utilities. read_configuration_file(dict_catalog["cases_file"], "case"),
               "syntactic_functions":utilities.read_syntactic_file (dict_catalog["syntactic_function_file"],"syntactic_function")
    }
    for i,f_corpus_path in enumerate(corpus_file_list):
        logging.info(f"generate_features::{f_corpus_path}")
        sentences_list = utilities.deserialize_file(f_corpus_path)
        dict_features = generate_pairs.pronominal_coreference_candidate_pairs(dict_info,sentences_list,f_corpus_path)
        add_category(dict_features,sentences_list,f_corpus_path)
        if i==0 :
            dict_features_names=coreference_features.get_feature_names()
            generate_feature_names_file(dict_features_names, feature_names_file)
            utilities.generate_scikit_learn_antet(dict_features_names, sklearn_file,training=1)
        utilities.append_scikit_learn_file(dict_features, sklearn_file)

def train () :
    """Train using the annotated corpus"""

    parser = argparse.ArgumentParser(
        description='Train the coreference model for your corpus')
    parser.add_argument("--catalog", type=str, help="The relative path to your language catalog of resources")
    parser.add_argument("--corpus_dir", type=str, help="The relative path to your language coreference annotated corpus")
    parser.add_argument("--sklearn_file", type=str,help="The relative path to store the trained model")
    parser.add_argument("--feature_names_file", type=str, help="The file containing the names of the computed features")

    args = parser.parse_args()

    if not (args.catalog and args.corpus_dir and args.sklearn_file and args.feature_names_file) :
        logging.error("Wrong arguments!")
        sys.exit()

    dict_catalog = utilities.read_resource_catalog(args.catalog)
    logging.info(f"train::Read Resource Catalog from=>{args.catalog}")

    corpus_file_list = utilities.read_corpus(args.corpus_dir,"*.conllu")
    logging.info(f"train::Read the traing corpus file names from=>{args.corpus_dir}")

    coreference_features.get_mention_global_scores (dict_catalog["global_mention_scores"])
    logging.info(f"""train::Read the global mention scores from=>{dict_catalog["global_mention_scores"]}""")

    coreference_features.get_eleri_abstractness (dict_catalog["eleri_abstractness"])
    logging.info(f"""train::Read Eleri Aedmaa abstractness scores from=> {dict_catalog["eleri_abstractness"]}""")

    coreference_features.init_embedding_models(dict_catalog["embeddings_file"], logging)
    logging.info(f"""train::Inited the embedding models from=> {dict_catalog["embeddings_file"]}""")

    logging.info(f"train::Train the corpus=>{args.corpus_dir}\n")
    generate_features(corpus_file_list, dict_catalog, args.sklearn_file,args.feature_names_file)
    logging.info(f"train::The trained model is in =>{args.sklearn_file}")
    logging.info(f"train::The computed coreference features names are in  =>{args.feature_names_file}")


def main():
    train()


if __name__ == '__main__':
    main()