#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = "Prerana Singhal"

import numpy as np
import theano, sys, os, gc
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
		print ("Usage: classify.py")
		print ("\t<model file path>")
		print ("\t<static word-vectors file path>")
		print ("\t<nonstatic word-vectors file path>")
		print ("\t<data file path (NO_FILE if do not want to classify data in a file)>")
		print ("\t<path(s) of 1 or more word vector files (NO_FILE if no wordvecfiles>")
		exit(0)

	model_file_path = sys.argv[1]
	static_file_path = sys.argv[2]
	nonstatic_file_path = sys.argv[3]
	fname = sys.argv[4]
	word_vec_files = sys.argv[5:]
	if word_vec_files[0]=='NO_FILE':
		word_vec_files=[]
	classifier, labels, threshold, preprocess, static_idx, nonstatic_idx = cPickle.load(open(model_file_path,"rb"))
	print('Model is loaded.')
	

	#loading word vectors
	static_Words = None
	Word_idx_map_static = None
	if static_idx>=0:
		word_vecs_static = load_vecs(vocab=[], dim=-1, filenames=[static_file_path]+word_vec_files, add_unknown = False, variance_random = 0, load_all=True)

		for wd in word_vecs_static.keys():
			for p in preprocess:
				temp_wd = p(wd)
			if temp_wd != wd:
				del word_vecs_static[wd]

		print('Static Word vectors are loaded for ' + str(len(word_vecs_static))+' words')

		Word_idx_map_static, Word_idxvec_matrix_static = get_word2vec_map(word_vecs=word_vecs_static, vocab=[], load_all=True)
		del word_vecs_static
		gc.collect()

		static_Words = theano.shared(value = np.asarray(Word_idxvec_matrix_static, dtype=theano.config.floatX), name = "static_Words")
		del Word_idxvec_matrix_static
		gc.collect()
	
	nonstatic_Words = None
	Word_idx_map_nonstatic = None
	if nonstatic_idx>=0:
		word_vecs_nonstatic = load_vecs(vocab=[], dim=-1, filenames=[nonstatic_file_path]+word_vec_files, add_unknown = False, variance_random = 0, load_all=True)
		print('Non-Static Word vectors are loaded for ' + str(len(word_vecs_nonstatic))+' words')

		for wd in word_vecs_nonstatic.keys():
			for p in preprocess:
				temp_wd = p(wd)
			if temp_wd != wd:
				del word_vecs_nonstatic[wd]

		Word_idx_map_nonstatic, Word_idxvec_matrix_nonstatic = get_word2vec_map(word_vecs=word_vecs_nonstatic, vocab=[], load_all=True)
		del word_vecs_nonstatic
		gc.collect()

		nonstatic_Words = theano.shared(value = np.asarray(Word_idxvec_matrix_nonstatic, dtype=theano.config.floatX), name = "nonstatic_Words")
		del Word_idxvec_matrix_nonstatic
		gc.collect()


	print('Model is being defined..')
	_, test_model = classifier.define_model(static_Words=static_Words, static_idx=static_idx, nonstatic_idx=nonstatic_idx, nonstatic_Words=nonstatic_Words)


	#classifying data from a file (each line has a data-point and nothing else)
	if fname != 'NO_FILE':
		print('The file ' + fname + ' is being classified..')
		data = open(fname,"rb").readlines()
		writer = csv.writer(open(fname+'_output.csv',"wb"), delimiter=',')
		writer.writerow(['Predicted Probabilities', 'Predicted labels', 'Processed string'])
		for text in data:
			result = classify(text=text, preprocess=preprocess, threshold=threshold, Word_idx_map_static=Word_idx_map_static, Word_idx_map_nonstatic=Word_idx_map_nonstatic, test_model=test_model)
			if threshold==1 or threshold==0:
				res_label = [labels[result[2]]]
			else:
				res_label = [labels[row] for row in range(len(result[2])) if result[2][row]==1]
			writer.writerow([result[1], res_label, result[0]])
		print('The output is stored in file : ' + fname+'_output.csv')


	#classifying data from console
	while(True):
		text = raw_input("\nEnter the text (space separated words) to classify:\n")
		result = classify(text=text, preprocess=preprocess, threshold=threshold, Word_idx_map_static=Word_idx_map_static, Word_idx_map_nonstatic=Word_idx_map_nonstatic, test_model=test_model)
		if threshold==1 or threshold==0:
			res_label = [labels[result[2]]]
		else:
			res_label = [labels[row] for row in range(len(result[2])) if result[2][row]==1]

		print ("Processed string : " + result[0])
		print ("Predicted probabilities (" + str(labels) + ") :: " + str(result[1]))
		print ("******************************************************************")
		print ("THE PREDICTED LABEL(S) FOR THE TEXT : " + str(res_label))
		print ("******************************************************************")



