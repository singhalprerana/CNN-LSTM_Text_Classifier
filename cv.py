#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = "Prerana Singhal"

import numpy as np
import theano, sys, os
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
	if len(sys.argv)<5:
		print ("Usage: cv.py")
		print ("\t<configuration file path>")
		print ("\t<network layers file path>")
		print ("\t<folder to store information file>")
		print ("\t<path(s) of 1 or more data files>")
		exit(0)

	config_file = sys.argv[1]
	layer_file = sys.argv[2]
	folder = sys.argv[3]
	data_files = sys.argv[4:]
	info_file_path = folder + '/cvinfo_' + str(datetime.now()).replace(' ','_').replace(':','-') + '.txt'
	print ('The information will be stored in file :   ' + info_file_path)

	# read configurations from file
	configs = load_configs(config_file)
	
	# reading layers from file
	layers, static_input, nonstatic_input = load_layers(layer_file)

	'''
	Extracting data from data files
	'''
	data_whole, labels = extract_data(filenames = data_files, preprocess = configs['preprocess'], delimiter=configs['delimiter'], labels_present=True)

	'''
	Calling cross_validation function
	'''
	print_status('\nCross-Validation information :', info_file_path)
	print_status('Configuration --> ' + str(configs), info_file_path)
	print_status('\nLabels are: ' + str(labels), info_file_path)
	
	print_status('Total number of data-points: ' + str(len(data_whole)), info_file_path)
	print_status('Number of cross-validation folds: ' + str(configs['cv_folds']), info_file_path)
	print_status('Number of cross-validation repeats: ' + str(configs['cv_repeats']), info_file_path)

	print_status('\nLayers of the Neural Network :', info_file_path)
	print_status(str(layers), info_file_path)
	
	cv(configs=configs, data_whole=data_whole, labels=labels, layers=layers, static_idx=static_input, nonstatic_idx=nonstatic_input, info_file_path=info_file_path)
	print ('The information is stored in file :   ' + info_file_path)




