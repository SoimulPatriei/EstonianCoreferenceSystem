#!/bin/bash

#This bash script trains on the corpus and then tests on new texts.
#First give the execution rights to the script: chmod u+x experiment_2.sh
#Run: ./experiment_2.sh



echo "Run the training script"
python train.py --catalog estonian_configuration_files/estonian_catalog.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file estonian_results/estonian_training_corpus-sklearn.txt  --feature_names_file estonian_results/estonian-computed-features.txt


echo "Run the test script"
python test.py --catalog estonian_configuration_files/estonian_catalog.xml --corpus_dir estonian_resources/estonian_test_corpus --corpus_tagged_dir estonian_resources/estonian_tagged_test_corpus --sklearn_file_training estonian_results/estonian_training_corpus-sklearn.txt --sklearn_file_test estonian_results/estonian_test_corpus-sklearn.txt --feature_names_file estonian_results/estonian-computed-features.txt --output_file coreference_pairs.txt





