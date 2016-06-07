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
	if len(sys.argv)<7:
		print ("Usage: testing.py")
		print ("\t<model file path>")
		print ("\t<static word-vectors file path>")
		print ("\t<nonstatic word-vectors file path>")
		print ("\t<data file path (csv file with delimiter=',')>")
		print ("\t<folder to store information and output files>")
		print ("\t<path(s) of 1 or more word vector files (NO_FILE if no wordvecfiles)>")
		exit(0)

	model_file_path = sys.argv[1]
	static_file_path = sys.argv[2]
	nonstatic_file_path = sys.argv[3]
	data_file = sys.argv[4]
	folder = sys.argv[5]
	word_vecs_files = sys.argv[6:]
	if word_vecs_files[0]=='NO_FILE':
		word_vecs_files=[]

	stamp = str(datetime.now()).replace(' ','_').replace(':','-')
	info_file_path = folder + '/testinginfo_' + stamp + '.txt'
	output_file_paths = [folder + '/testingoutput_' + stamp + '.csv', folder + '/testingmisclassification_' + stamp + '.csv']
	print ('The information will be stored in file  :  ' + info_file_path)

	'''
	Extracting data from data files
	'''
	preprocess = cPickle.load(open(model_file_path,"rb"))[3]
	data, labels = extract_data(filenames = [data_file], preprocess = preprocess, delimiter=',', labels_present=True)


	'''
	Calling testing function
	'''	
	print_status('\nTesting information :', info_file_path)
	print_status('\nLabels are: ' + str(labels), info_file_path)
	
	testing(data=data, threshold=-1, word_vec_files=word_vecs_files, model_file_path=model_file_path, nonstatic_file_path=nonstatic_file_path, static_file_path=static_file_path, info_file_path=info_file_path, output_file_paths=output_file_paths)
	print ('The information is stored in file                :  ' + info_file_path)
	print ('The whole output is stored in file               :  ' + output_file_paths[0])
	print ('The misclassification output is stored in file   :  ' + output_file_paths[1])




