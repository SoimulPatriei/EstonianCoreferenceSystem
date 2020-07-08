#!/bin/bash

#This bash script runs the training program with the default arguments
#First give the execution rights to the script: chmod u+x train.sh
#Run: ./train.sh

echo "Run the training script"
python train.py --catalog estonian_configuration_files/estonian_catalog.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file estonian_results/estonian_training_corpus-sklearn.txt  --feature_names_file estonian_results/estonian-computed-features.txt