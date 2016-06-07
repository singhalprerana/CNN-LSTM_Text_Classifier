#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = "Prerana Singhal"

import numpy as np
import theano, sys, os, csv
import theano.tensor as T
import cPickle
from random import shuffle
from datetime import datetime

from dataset_preparation import *
from neural_net_classes import *
from classification import *

import warnings
warnings.filterwarnings("ignore")



if __name__=="__main__":
	if len(sys.argv)<9:
		print ("Usage: training.py")
		print ("\t<configuration file path>")
		print ("\t<network layers file path>")
		print ("\t<existing training model file path (NO_MODEL if do not want to load model)>")
		print ("\t<existing static word-vectors file path (NO_STATIC if do not want to load model)>")
		print ("\t<existing nonstatic word-vectors file path (NO_NONSTATIC if do not want to load model)>")
		print ("\t<folder to store information and model files>")
		print ("\t<validation data file path (NO_VALIDATION_FILE if do not have such file)>")
		print ("\t<path(s) of 1 or more data files>")
		exit(0)

	config_file = sys.argv[1]
	layer_file = sys.argv[2]
	existing_model_file = sys.argv[3] if sys.argv[3]!='NO_MODEL' else ''
	existing_static_file = sys.argv[4] if sys.argv[4]!='NO_STATIC' else ''
	existing_nonstatic_file = sys.argv[5] if sys.argv[5]!='NO_NONSTATIC' else ''
	folder = sys.argv[6]
	validation_file = sys.argv[7] if sys.argv[7]!='NO_VALIDATION_FILE' else ''
	data_files = sys.argv[8:]

	stamp = str(datetime.now()).replace(' ','_').replace(':','-')
	info_file_path = folder + '/traininginfo_' + stamp + '.txt'
	model_file_path = folder + '/trainingmodel_' + stamp + '.p'
	nonstatic_file_path = folder + '/trainingwordvecs_nonstatic_' + stamp + '.p'
	static_file_path = folder + '/trainingwordvecs_static_' + stamp + '.p'
	print ('The information will be stored in file  :  ' + info_file_path)

	# read configurations from file
	configs = load_configs(config_file)
	
	# reading layers from file
	layers, static_input, nonstatic_input = load_layers(layer_file)
	
	'''
	Extracting data from data files
	'''
	data, labels = extract_data(filenames = data_files, preprocess = configs['preprocess'], delimiter=configs['delimiter'], labels_present=True)
	if validation_file!='':
		validation_data, _ = extract_data(filenames = [validation_file], preprocess = configs['preprocess'], delimiter=configs['delimiter'], labels_present=True)
	else:
		validation_data = []

	'''
	Calling training function
	'''	
	print_status('\nTraining information :', info_file_path)
	print_status('Configuration --> ' + str(configs), info_file_path)
	print_status('\nLabels are: ' + str(labels), info_file_path)
	print_status('Total number of data-points: ' + str(len(data)), info_file_path)

	print_status('\nLayers of the Neural Network :', info_file_path)
	print_status('static_input : ' + str(static_input) + ' , nonstatic_input : ' + str(nonstatic_input), info_file_path)
	print_status(str(layers), info_file_path)
	
	training(configs=configs, existing_model_file=existing_model_file, existing_nonstatic_file=existing_nonstatic_file, validation_data=validation_data, layers=layers, static_idx=static_input, nonstatic_idx=nonstatic_input, data=data, labels=labels, model_file_path=model_file_path, info_file_path=info_file_path, nonstatic_file_path=nonstatic_file_path, static_file_path=static_file_path)
	print ('The information is stored in file  :  ' + info_file_path)
	print ('The model is stored in file        :  ' + model_file_path)
	if nonstatic_input>=0:
		print ('The nonstatic word-vecs are stored in file        :  ' + nonstatic_file_path)
	if static_input>=0:
		print ('The static word-vecs are stored in file        :  ' + static_file_path)




