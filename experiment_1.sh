#!/bin/bash

#This bash script runs the training program for each pronoun and then applies stratified cross reference to the trained model.  
#First give the execution rights to the script: chmod u+x experiment_1.sh
#Run: ./experiment_1.sh

echo "Train the models for each pronoun"

echo "mina model"
python train.py --catalog Experiment_I/Configuration/estonian_catalog_mina.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file Experiment_I/Models/mina_model.txt  --feature_names_file estonian_results/estonian-computed-features.txt


echo "tema model"
python train.py --catalog Experiment_I/Configuration/estonian_catalog_tema.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file Experiment_I/Models/tema_model.txt  --feature_names_file estonian_results/estonian-computed-features.txt


echo "see model"
python train.py --catalog Experiment_I/Configuration/estonian_catalog_see.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file Experiment_I/Models/see_model.txt  --feature_names_file estonian_results/estonian-computed-features.txt

echo "kes model"
python train.py --catalog Experiment_I/Configuration/estonian_catalog_kes.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file Experiment_I/Models/kes_model.txt  --feature_names_file estonian_results/estonian-computed-features.txt


echo "mis model"
python train.py --catalog Experiment_I/Configuration/estonian_catalog_mis.xml --corpus_dir estonian_resources/estonian_training_corpus --sklearn_file Experiment_I/Models/mis_model.txt  --feature_names_file estonian_results/estonian-computed-features.txt




echo "----------------Stratified Cross Validation for each model-------------------------"
echo "mina SKV"

python stratified_cross_validation.py  --model Experiment_I/Models/mina_model.txt --feature_names_file estonian_results/estonian-computed-features.txt --logging_file Experiment_I/Results/mina_results.txt


echo "tema SKV"
python stratified_cross_validation.py  --model Experiment_I/Models/tema_model.txt --feature_names_file estonian_results/estonian-computed-features.txt --logging_file Experiment_I/Results/tema_results.txt

echo "see SKV"
python stratified_cross_validation.py  --model Experiment_I/Models/see_model.txt --feature_names_file estonian_results/estonian-computed-features.txt --logging_file Experiment_I/Results/see_results.txt

echo "kes SKV"
python stratified_cross_validation.py  --model Experiment_I/Models/kes_model.txt --feature_names_file estonian_results/estonian-computed-features.txt --logging_file Experiment_I/Results/kes_results.txt


echo "mis SKV"
python stratified_cross_validation.py  --model Experiment_I/Models/mis_model.txt --feature_names_file estonian_results/estonian-computed-features.txt --logging_file Experiment_I/Results/mis_results.txt








