#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
CODE FOR FOR CLASSIFICATION OF TEXT USING NEURAL NETWORKS
Some of the code is modified from
- https://github.com/yoonkim/CNN_sentence (Convolutional Neural Networks for Sentence Classification)
- deeplearning.net/tutorial (for ConvNet and LSTM classes)
"""
from __future__ import print_function
__author__ = "Prerana Singhal"

import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import sys, csv, re, os
from random import shuffle
from datetime import datetime

from dataset_preparation import *
from neural_net_classes import *

import warnings
warnings.filterwarnings("ignore")



"""
Class for Neural Network Classifier
"""
class Network_Classifier(object):
	"""
	This represents a neural network consisting of 'layers' (FullyConnected, LSTM, Convolution, etc)
	configs :  Different configurations of the neural network
	network_layers : list of layers of network with their configurations and input-output sequence;
	"""
	
	def __init__(self, network_layers, configs, num_labels): 	
		# configs are the configurations (all are strings)
		
		# some integer for random state generation
		if 'random_number' not in configs:	
			configs['random_number'] = 9876
		self.rng = np.random.RandomState(int(configs['random_number']))

		# dropout rate between 0 and 1
		if 'dropout' not in configs:	
			configs['dropout'] = 0
		self.dropout = float(configs['dropout'])

		# update (back-propagation) function : gradient or adadelta
		if 'update_function' not in configs:	
			configs['update_function'] = '_updates_gradient'
		self.update = eval(configs['update_function'].lower())

		# cost (error) function : squared-error or negative-log-likelihood
		if 'error_function' not in configs:	
			configs['error_function'] = '_square_error'
		self.error = eval(configs['error_function'].lower())

		if 'last_activation' not in configs:
			last_activation = 'Softmax'
		else:
			last_activation = configs['last_activation'].capitalize()

		# initialize the layers of the network
		self.network_layers = []
		self.params = []
		for layer in network_layers:
			configs = layer[1]
			
			if layer[0] == 'LSTM' :
				dim_in = int(configs['dim_in'])
				if configs['dim_out'] == 'labels':
					configs['dim_out'] = num_labels
				dim_out = int(configs['dim_out'])
				if 'pooling' in configs:
					pooling = eval(configs['pooling'].capitalize())
				else:
					pooling = Mean_pooling
				if 'window' in configs:
					window = int(configs['window'])
				else:
					window = 1
				if 'use_bias' in configs:
					use_bias = eval(configs['use_bias'].capitalize())
				else:
					use_bias = True
				if 'use_last_output' in configs:
					use_last_output = eval(configs['use_last_output'].capitalize())
				else:
					use_last_output = False
				layerx = LSTMLayer(rng = self.rng, dim_in = dim_in, dim_out = dim_out, window = window, pooling = pooling, Wx=[None,None,None,None], Wh=[None,None,None,None], b=[None,None,None,None], use_bias = use_bias, use_last_output = use_last_output)
			
			elif layer[0] == 'Convolution' :
				dim_in = int(configs['dim_in'])
				if configs['dim_out'] == 'labels':
					configs['dim_out'] = num_labels
				dim_out = int(configs['dim_out'])
				if 'activation' in configs:
					activation = eval(configs['activation'].capitalize())
				else:
					activation = Sigmoid
				if 'pooling' in configs:
					pooling = eval(configs['pooling'].capitalize())
				else:
					pooling = Max_pooling
				if 'window' in configs:
					window = int(configs['window'])
				else:
					window = 3
				if 'use_bias' in configs:
					use_bias = eval(configs['use_bias'].capitalize())
				else:
					use_bias = True
				layerx = ConvolutionLayer(rng = self.rng, dim_in = dim_in, dim_out = dim_out, window = window, activation = activation, pooling = pooling, W=None, b=None, use_bias = use_bias)
			
			elif layer[0] == 'FullyConnected' :
				n_in = int(configs['n_in'])
				if configs['n_out'] == 'labels':
					configs['n_out'] = num_labels
				n_out = int(configs['n_out'])
				if 'activation' in configs:
					if configs['activation'] == 'last':
						activation = eval(last_activation)
					else:
						activation = eval(configs['activation'].capitalize())
				else:
					activation = Sigmoid
				if 'pooling' in configs:
					pooling = eval(configs['pooling'].capitalize())
				else:
					pooling = None
				if 'use_bias' in configs:
					use_bias = eval(configs['use_bias'].capitalize())
				else:
					use_bias = True
				layerx = FullyConnectedLayer(rng = self.rng, n_in = n_in, n_out = n_out, activation = activation, pooling = pooling, W=None, b=None, use_bias = use_bias)
			
			self.network_layers.append([layerx, layer[2], layer[3]])
			self.params += layerx.params


	def define_model(self, static_Words, static_idx, nonstatic_Words, nonstatic_idx):
		"""
		Function to construct the theano functions for training, validating and testing
		"""
		y = T.ivector('y')
		rho = T.scalar('rho')
		iodict = {}
		dropout_iodict = {}

		if static_idx>=0:
			x_static = T.ivector('x_static')
			input_static = static_Words[x_static]
			iodict[static_idx] = input_static
			dropout_iodict[static_idx] = input_static
			final_output = [static_idx, input_static, None]
		
		if nonstatic_idx>=0:
			x_nonstatic = T.ivector('x_nonstatic')
			input_nonstatic = nonstatic_Words[x_nonstatic]
			iodict[nonstatic_idx] = input_nonstatic
			dropout_iodict[nonstatic_idx] = input_nonstatic
			final_output = [nonstatic_idx, input_nonstatic, None]
		
		for layer in self.network_layers:
			iodict[layer[2]] = layer[0].predict(input = T.concatenate([iodict[i] for i in layer[1]]))
			dropout_iodict[layer[2]] = layer[0].predict_dropout(input = T.concatenate([dropout_iodict[i] for i in layer[1]]), rng = self.rng, p = self.dropout)
			if final_output[0] < layer[2]:
				final_output = [layer[2], layer[1], layer[0]]

		dropout_iodict[final_output[0]] = final_output[2].predict(input = T.concatenate([dropout_iodict[i] for i in final_output[1]]))
		
		output = iodict[final_output[0]]
		cost = self.error(output = output.flatten(), act_y = y)
		dropout_cost = self.error(output = dropout_iodict[final_output[0]].flatten(), act_y = y)
		if nonstatic_idx>=0:
			nonstatic_Words.name = 'NON_STATIC_INPUT'
			grad_updates = self.update(params = self.params + [nonstatic_Words], cost=dropout_cost, rho=T.cast(rho,dtype=theano.config.floatX))
		else:
			grad_updates = self.update(params=self.params, cost=dropout_cost, rho=T.cast(rho,dtype=theano.config.floatX))

		if static_idx>=0 and nonstatic_idx>=0:
			train_model = theano.function([x_static, x_nonstatic, y, rho], cost, updates=grad_updates)
			test_model = theano.function([x_static, x_nonstatic], output)
		elif static_idx>=0:
			train_model = theano.function([x_static, y, rho], cost, updates=grad_updates)
			test_model = theano.function([x_static], output)
		elif nonstatic_idx>=0:
			train_model = theano.function([x_nonstatic, y, rho], cost, updates=grad_updates)
			test_model = theano.function([x_nonstatic], output)

		return train_model, test_model


	def __getstate__(self):
		return (self.network_layers, self.rng, self.dropout, self.update, self.error)

	def __setstate__(self, state):
		self.network_layers, self.rng, self.dropout, self.update, self.error = state
		self.params = []
		for layer in self.network_layers:
			self.params += layer[0].params



"""
Function to print in console and in file
"""
def print_status(str, fname):
	print(str)
	print(str, file=open(fname,'ab'))


"""
Function to load info about layers of the network from a file and store in a list
"""
def load_layers(layer_file):
	layers=[]
	with open(layer_file,"rb") as f:
		reader=csv.reader(f,delimiter=",")
		lines=[]
		for row in reader:
			lines.append(row)
		static_input = int(lines[0][1])		#first row in file
		nonstatic_input = int(lines[1][1])	#second row in file

		for row in range(2,len(lines),3):
			#configurations of the layer
			config_layer = {}
			for i in range(len(lines[row+1])):
				config_layer[lines[row+1][i]] = lines[row+2][i]
			#input to layer
			inp = [int(num) for num in lines[row][1].split(',')]
			layers.append([lines[row][0], config_layer, inp, int(lines[row][2])])
	
	return layers, static_input, nonstatic_input


"""
Function to load configurations from file and assign default values if necessary
"""
def load_configs(config_file):
	configs={}
	with open(config_file,"rb") as f:
		reader=csv.reader(f,delimiter=",")
		for row in reader:
			configs[row[0]]=row[1]

	'''
	Assigning default config values if not present in file
	'''
	configs['epochs'] = 25 if 'epochs' not in configs else int(configs['epochs'])
	configs['epsilon'] = 0.1 if 'epsilon' not in configs else float(configs['epsilon'])
	# validation fraction of data if explicit validation data file is not specified; 0 means no validation set during training 
	configs['validation'] = 0.1 if 'validation' not in configs else float(configs['validation'])     
	
	#prediction threshold : 0 implies choose the output neuron with minimum value
	#					    1 implies choose the output neuron with maximum value
	#					    float value is the threshold to decide between 0 and 1
	configs['threshold'] = 1 if 'threshold' not in configs else float(configs['threshold'])
	
	configs['random_number'] = 9876 if 'random_number' not in configs else int(configs['random_number'])
	configs['dropout'] = 0.4 if 'dropout' not in configs else float(configs['dropout'])
	configs['learning_rate'] = 0.95 if 'learning_rate' not in configs else float(configs['learning_rate'])
	configs['update_function'] = 'updates_gradient' if 'update_function' not in configs else configs['update_function'].lower()
	configs['error_function'] = 'square_error' if 'error_function' not in configs else configs['error_function'].lower()

	#preprocessing functions in the desired order
	configs['preprocess'] = [replace_accents, to_lowercase] if 'preprocess' not in configs else [eval(fn.lower()) for fn in configs['preprocess'].split(',') if fn!='']  
	#field delimiter in data_files
	configs['delimiter'] = ',' if 'delimiter' not in configs else configs['delimiter']

	configs['wordvec_files'] = ['GoogleNews-vectors-negative300.bin'] if 'wordvec_files' not in configs else configs['wordvec_files'].split(',')
	configs['variance_random'] = 0.25 if 'variance_random' not in configs else float(configs['variance_random'])
	configs['dim'] = 300 if 'dim' not in configs else int(configs['dim'])

	configs['cv_folds'] = 5 if 'cv_folds' not in configs else int(configs['cv_folds'])
	configs['cv_repeats'] = 2 if 'cv_repeats' not in configs else int(configs['cv_repeats'])

	configs['last_activation'] = 'Softmax' if 'last_activation' not in configs else configs['last_activation']
	return configs


"""
Function to calculate fscore and accuracy values
"""
def scoring(prob_pred, Ytest, threshold, labels):
	fscores=[]
	if threshold == 1 or threshold == 0:
		if threshold == 1:
			y_pred = prob_pred.argmax(axis=1)
			Ytest = Ytest.argmax(axis=1)
		else:
			y_pred = prob_pred.argmin(axis=1)
			Ytest = Ytest.argmin(axis=1)
		correct = np.equal(Ytest,y_pred).sum()
		test_accuracy = (float(correct)/(len(Ytest))) * 100		
			
		for l in range(len(labels)):
			tpx = sum([(Ytest[i]==l) and (y_pred[i]==l) for i in range(len(Ytest))])
			tnx = sum([(Ytest[i]!=l) and (y_pred[i]!=l) for i in range(len(Ytest))])
			fpx = sum([(Ytest[i]!=l) and (y_pred[i]==l) for i in range(len(Ytest))])
			fnx = sum([(Ytest[i]==l) and (y_pred[i]!=l) for i in range(len(Ytest))])
			fscores.append([tpx,tnx,fpx,fnx,(200.0*tpx)/(2.0*tpx+fpx+fnx)])

		return test_accuracy, fscores, y_pred, Ytest
	
	else:
		y_pred = prob_pred >= threshold
		y_pred = y_pred.astype(int)
		correct = np.equal(Ytest,y_pred).sum()
		test_accuracy = (float(correct)/(len(Ytest)*len(labels))) * 100
		for j in range(len(labels)):
			tpx=sum([(Ytest[i][j]==1) and (y_pred[i][j]==1) for i in range(len(Ytest))])
			tnx=sum([(Ytest[i][j]==0) and (y_pred[i][j]==0) for i in range(len(Ytest))])
			fpx=sum([(Ytest[i][j]==0) and (y_pred[i][j]==1) for i in range(len(Ytest))])
			fnx=sum([(Ytest[i][j]==1) and (y_pred[i][j]==0) for i in range(len(Ytest))])
			if 2*tpx+fpx+fnx==0:
				fscores.append([tpx,tnx,fpx,fnx,0])
			else:
				fscores.append([tpx,tnx,fpx,fnx,(200.0*tpx)/(2.0*tpx+fpx+fnx)])

		SuG=((Ytest==y_pred)*(Ytest==1)).sum()
		G=Ytest.sum()
		S=y_pred.sum()
		P=float(SuG)/float(S)
		R=float(SuG)/float(G)
		F1_measure=(200*P*R)/(P+R)

		return test_accuracy, fscores, y_pred, Ytest, F1_measure


"""
Function for training on a dataset_preparation with (or without or random) validation
"""
def training(configs, existing_model_file, existing_nonstatic_file, validation_data, layers, static_idx, nonstatic_idx, data, labels, model_file_path, info_file_path, nonstatic_file_path, static_file_path):
	rho = configs['learning_rate']
	vocab = get_vocab(data)
	num_data = len(data)

	if validation_data==[]:
		split_point = num_data - int(configs['validation'] * num_data)	# for validation
		num_valid = num_data - split_point
		print_status('\nNumber of training data-points: ' + str(split_point), info_file_path)
		if configs['validation']>0:		# if validation is to be applied
			print_status('Number of validation data-points (randomly chosen in each epoch): ' + str(num_valid), info_file_path)
	else:
		num_valid = len(validation_data)
		vocab_valid = get_vocab(validation_data)
		print_status('\nNumber of training data-points: ' + str(num_data), info_file_path)
		print_status('\nNumber of validation data-points: ' + str(num_valid), info_file_path)
		vocab = list(set(vocab+vocab_valid))
	print_status('Training-Validation vocabulary size: ' + str(len(vocab)), info_file_path)

	#initialising model..
	if existing_model_file!='':
		classifier, _, _, _, static_idx, nonstatic_idx = cPickle.load(open(existing_model_file,"rb"))
	else:
		classifier = Network_Classifier(network_layers=layers, configs = {'random_number' : configs['random_number'], 'dropout' : configs['dropout'], 'update_function' : configs['update_function'], 'error_function' : configs['error_function']}, num_labels = len(labels))


	#loading word vectors
	static_Words = None
	if static_idx>=0:
		word_vecs_static = load_vecs(vocab=vocab, dim=configs['dim'], filenames=[existing_nonstatic_file]+configs['wordvec_files'], add_unknown = True, variance_random = configs['variance_random'])
		print_status('Static Word vectors are loaded for ' + str(len(word_vecs_static))+' words', info_file_path)

		Word_idx_map_train_static, Word_idxvec_matrix_train_static = get_word2vec_map(word_vecs=word_vecs_static, vocab=vocab)
		Xdata1_static, Ydata1 = make_idx_data_cv(data=data, Word_idx_map=Word_idx_map_train_static, labels_present=True)
		if validation_data!=[]:
			Xvalid_static, Yvalid = make_idx_data_cv(data=validation_data, Word_idx_map=Word_idx_map_train_static, labels_present=True)
		static_Words = theano.shared(value = np.asarray(Word_idxvec_matrix_train_static, dtype=theano.config.floatX), name = "static_Words")
	
	nonstatic_Words = None
	if nonstatic_idx>=0:
		if existing_nonstatic_file!='':
			word_vecs_nonstatic = load_vecs(vocab=vocab, dim=configs['dim'], filenames=[existing_nonstatic_file]+configs['wordvec_files'], add_unknown = True, variance_random = configs['variance_random'])
		else:
			word_vecs_nonstatic = load_vecs(vocab=vocab, dim=configs['dim'], filenames=configs['wordvec_files'], add_unknown = True, variance_random = configs['variance_random'])
		print_status('Non-Static Word vectors are loaded for ' + str(len(word_vecs_nonstatic))+' words', info_file_path)

		Word_idx_map_train_nonstatic, Word_idxvec_matrix_train_nonstatic = get_word2vec_map(word_vecs=word_vecs_nonstatic, vocab=vocab)
		Xdata1_nonstatic, Ydata1 = make_idx_data_cv(data=data, Word_idx_map=Word_idx_map_train_nonstatic, labels_present=True)
		if validation_data!=[]:
			Xvalid_nonstatic, Yvalid = make_idx_data_cv(data=validation_data, Word_idx_map=Word_idx_map_train_nonstatic, labels_present=True)
		nonstatic_Words = theano.shared(value = np.asarray(Word_idxvec_matrix_train_nonstatic, dtype=theano.config.floatX), name = "nonstatic_Words")
	
	
	train_model, test_model = classifier.define_model(static_Words=static_Words, static_idx=static_idx, nonstatic_idx=nonstatic_idx, nonstatic_Words=nonstatic_Words)
	print('Model is defined; Training is started')


	if static_idx>=0:
		cPickle.dump(word_vecs_static, open(static_file_path, "wb"))
	temp_file_path = nonstatic_file_path + "_temp.p"


	least_cost = 1000
	best_accuracy = -1
	indices = range(num_data)

	for epoch in range(configs['epochs']):
		shuffle(indices)
		Ydata = Ydata1[indices]
		if static_idx>=0:
			Xdata_static = Xdata1_static[indices]
		if nonstatic_idx>=0:
			Xdata_nonstatic = Xdata1_nonstatic[indices]
		

		if validation_data==[]:
			if static_idx>=0:
				Xvalid_static = Xdata_static[indices[split_point:]]
				Xtrain_static = Xdata_static[indices[:split_point]]
			if nonstatic_idx>=0:
				Xvalid_nonstatic = Xdata_nonstatic[indices[split_point:]]
				Xtrain_nonstatic = Xdata_nonstatic[indices[:split_point]]
			Ytrain = Ydata[indices[:split_point]]
			Yvalid = Ydata[indices[split_point:]]
		else:
			if static_idx>=0:
				Xtrain_static = Xdata_static
			if nonstatic_idx>=0:
				Xtrain_nonstatic = Xdata_nonstatic
			Ytrain = Ydata


		cost = 0
		for i in range(len(Ytrain)):
			print(str(i+1),end='\r')
			sys.stdout.flush()

			if static_idx>=0 and nonstatic_idx>=0:
				cost_epoch = train_model(Xtrain_static[i], Xtrain_nonstatic[i], Ytrain[i], rho)
			elif static_idx>=0:
				cost_epoch = train_model(Xtrain_static[i], Ytrain[i], rho)
			elif nonstatic_idx>=0:
				cost_epoch = train_model(Xtrain_nonstatic[i], Ytrain[i], rho)
			cost += cost_epoch
		cost = cost/len(Ytrain)

		if validation_data!=[] or configs['validation']>0:
			outp = []
			for i in range(len(Yvalid)):
				if static_idx>=0 and nonstatic_idx>=0:
					outp += list(test_model(Xvalid_static[i], Xvalid_nonstatic[i]))
				elif static_idx>=0:
					outp += list(test_model(Xvalid_static[i]))
				elif nonstatic_idx>=0:
					outp += list(test_model(Xvalid_nonstatic[i]))
			
			outp = np.asarray(outp)

			xs = scoring(prob_pred=outp, Ytest=Yvalid, threshold=configs['threshold'], labels=labels)
			test_accuracy, fscores = xs[0], xs[1]
			valid_fscore = np.asarray([row[-1] for row in fscores]).mean()

			print_status('Epoch ' + str(epoch+1) + ' \t:: Training error : ' + str(round(cost,9)) + ' \t:: Validation fscore : ' + str(round(valid_fscore,3)) + '%', info_file_path)
			
			if best_accuracy < valid_fscore:
				best_accuracy = valid_fscore
				least_cost = cost
				cPickle.dump([classifier, labels, configs['threshold'], configs['preprocess'], static_idx, nonstatic_idx], open(model_file_path, "wb"))
				if nonstatic_idx>=0:
					cPickle.dump(nonstatic_Words.get_value(), open(temp_file_path, "wb"))
					Word_idxvec_matrix_train_nonstatic = nonstatic_Words.get_value()
					word_vecs_nonstatic = {}
					for wd in Word_idx_map_train_nonstatic:
						word_vecs_nonstatic[wd] = Word_idxvec_matrix_train_nonstatic[Word_idx_map_train_nonstatic[wd]]
					cPickle.dump(word_vecs_nonstatic, open(nonstatic_file_path, "wb"))

			
			elif best_accuracy == valid_fscore and least_cost > cost:
				least_cost = cost
				cPickle.dump([classifier, labels, configs['threshold'], configs['preprocess'], static_idx, nonstatic_idx], open(model_file_path, "wb"))
				if nonstatic_idx>=0:
					cPickle.dump(nonstatic_Words.get_value(), open(temp_file_path, "wb"))
					Word_idxvec_matrix_train_nonstatic = nonstatic_Words.get_value()
					word_vecs_nonstatic = {}
					for wd in Word_idx_map_train_nonstatic:
						word_vecs_nonstatic[wd] = Word_idxvec_matrix_train_nonstatic[Word_idx_map_train_nonstatic[wd]]
					cPickle.dump(word_vecs_nonstatic, open(nonstatic_file_path, "wb"))

			elif best_accuracy > valid_fscore and cost - least_cost > configs['epsilon']:
				rho = rho - rho * configs['epsilon']
				print('Learning rate reduced to '+str(rho))
				classifier, labels, configs['threshold'], configs['preprocess'], static_idx, nonstatic_idx = cPickle.load(open(model_file_path,"rb"))
				if nonstatic_idx>=0:
					nonstatic_Words = theano.shared(value = np.asarray(cPickle.load(open(temp_file_path,"rb")), dtype=theano.config.floatX), name = "nonstatic_Words")
				train_model, test_model = classifier.define_model(static_Words=static_Words, static_idx=static_idx, nonstatic_idx=nonstatic_idx, nonstatic_Words=nonstatic_Words)


		else:
			print_status('Epoch ' + str(epoch+1) + ' :: Training error : ' + str(round(cost,9)), info_file_path)
			
			if cost <= least_cost:
				least_cost = cost
				cPickle.dump([classifier, labels, configs['threshold'], configs['preprocess'], static_idx, nonstatic_idx], open(model_file_path, "wb"))
				if nonstatic_idx>=0:
					cPickle.dump(nonstatic_Words.get_value(), open(temp_file_path, "wb"))
					Word_idxvec_matrix_train_nonstatic = nonstatic_Words.get_value()
					word_vecs_nonstatic = {}
					for wd in Word_idx_map_train_nonstatic:
						word_vecs_nonstatic[wd] = Word_idxvec_matrix_train_nonstatic[Word_idx_map_train_nonstatic[wd]]
					cPickle.dump(word_vecs_nonstatic, open(nonstatic_file_path, "wb"))

			elif cost - least_cost > configs['epsilon']:
				rho = rho - rho * configs['epsilon']				
				print('Learning rate reduced to '+str(rho))
				classifier, labels, configs['threshold'], configs['preprocess'], static_idx, nonstatic_idx = cPickle.load(open(model_file_path,"rb"))
				if nonstatic_idx>=0:
					nonstatic_Words = theano.shared(value = np.asarray(cPickle.load(open(temp_file_path,"rb")), dtype=theano.config.floatX), name = "nonstatic_Words")
				train_model, test_model = classifier.define_model(static_Words=static_Words, static_idx=static_idx, nonstatic_idx=nonstatic_idx, nonstatic_Words=nonstatic_Words)
	os.remove(temp_file_path)


"""
Function for testing on a dataset
"""
def testing(data, threshold, word_vec_files, model_file_path, nonstatic_file_path, static_file_path, info_file_path, output_file_paths=[]):
	xx = cPickle.load(open(model_file_path,"rb"))
	classifier, labels, static_idx, nonstatic_idx = xx[0], xx[1], xx[4], xx[5]
	if threshold<0:		# when not passed as argument, take the value from the trained model
		threshold = xx[2]

	vocab = get_vocab(data)

	print_status('\nNumber of test data-points: ' + str(len(data)), info_file_path)
	print_status('Test vocabulary size: ' + str(len(vocab)), info_file_path)


	#loading word vectors
	static_Words = None
	if static_idx>=0:
		word_vecs_static = load_vecs(vocab=vocab, dim=-1, filenames=[static_file_path]+word_vec_files, add_unknown = False, variance_random = 0)
		print_status('Static Word vectors are loaded for ' + str(len(word_vecs_static))+' words', info_file_path)

		Word_idx_map_train_static, Word_idxvec_matrix_train_static = get_word2vec_map(word_vecs=word_vecs_static, vocab=vocab)
		Xtest_static, Ytest = make_idx_data_cv(data=data, Word_idx_map=Word_idx_map_train_static, labels_present=True)
		static_Words = theano.shared(value = np.asarray(Word_idxvec_matrix_train_static, dtype=theano.config.floatX), name = "static_Words")
	
	nonstatic_Words = None
	if nonstatic_idx>=0:
		word_vecs_nonstatic = load_vecs(vocab=vocab, dim=-1, filenames=[nonstatic_file_path]+word_vec_files, add_unknown = False, variance_random = 0)
		print_status('Non-Static Word vectors are loaded for ' + str(len(word_vecs_nonstatic))+' words', info_file_path)

		Word_idx_map_train_nonstatic, Word_idxvec_matrix_train_nonstatic = get_word2vec_map(word_vecs=word_vecs_nonstatic, vocab=vocab)
		Xtest_nonstatic, Ytest = make_idx_data_cv(data=data, Word_idx_map=Word_idx_map_train_nonstatic, labels_present=True)
		nonstatic_Words = theano.shared(value = np.asarray(Word_idxvec_matrix_train_nonstatic, dtype=theano.config.floatX), name = "nonstatic_Words")


	_, test_model = classifier.define_model(static_Words=static_Words, static_idx=static_idx, nonstatic_idx=nonstatic_idx, nonstatic_Words=nonstatic_Words)
	print('Model is loaded and defined; Testing is being done')


	prob_pred = []
	for i in range(len(Ytest)):
		print(str(i+1),end='\r')
		sys.stdout.flush()
		if static_idx>=0 and nonstatic_idx>=0:
			prob_pred += list(test_model(Xtest_static[i], Xtest_nonstatic[i]))
		elif static_idx>=0:
			prob_pred += list(test_model(Xtest_static[i]))
		elif nonstatic_idx>=0:
			prob_pred += list(test_model(Xtest_nonstatic[i]))
	prob_pred = np.asarray(prob_pred)
	
	if threshold == 1 or threshold == 0:
		test_accuracy, fscores, y_pred, Ytest = scoring(prob_pred=prob_pred, Ytest=Ytest, threshold=threshold, labels=labels)
	else:
		test_accuracy, fscores, y_pred, Ytest, F1_measure = scoring(prob_pred=prob_pred, Ytest=Ytest, threshold=threshold, labels=labels)

	print('         ')
	print_status('\nTESTING ACCURACY : ' + str(round(test_accuracy,3)) + '%', info_file_path)
	if threshold != 1 and threshold != 0:
		print_status('TESTING F1-MEASURE : ' + str(round(F1_measure,3)) + '%', info_file_path)
	for i in range(len(labels)):
		print_status('Label ' + labels[i] + ' :: \tFscore : ' + str(round(fscores[i][-1],3)) + '% \t:: \tTP:' + str(fscores[i][0]) + ' ,\tTN:' + str(fscores[i][1]) + ' ,\tFP:' + str(fscores[i][2]) + ' ,\tFN:' + str(fscores[i][3]), info_file_path)
	avg_fscore = np.asarray([row[-1] for row in fscores]).mean()
	print_status('AVERAGE FSCORE : ' + str(round(avg_fscore,3)) + '%\n', info_file_path)


	if output_file_paths!=[]:
		outpf = csv.writer(open(output_file_paths[0],"wb"), delimiter=',')
		outpf.writerow(['Probabilities('+str(labels)+')','Predicted label(s)','Actual label(s)','Processed text'])
		if len(output_file_paths)>1:		#misclassification file
			misf = csv.writer(open(output_file_paths[1],"wb"), delimiter=',')
			misf.writerow(['Output no.','Probabilities('+str(labels)+')','Predicted label(s)','Actual label(s)','Processed text'])
		
		for i in range(len(data)):
			if threshold==0 or threshold==1:
				outpf.writerow([prob_pred[i], labels[y_pred[i]], labels[Ytest[i]], data[i][0]])
				if len(output_file_paths)>1 and y_pred[i] != Ytest[i]:
					misf.writerow([i+2, prob_pred[i], labels[y_pred[i]], labels[Ytest[i]], data[i][0]])
			else:
				yp = [labels[row] for row in range(len(y_pred[i])) if y_pred[i][row]==1]
				ya = [labels[row] for row in range(len(Ytest[i])) if Ytest[i][row]==1]
				outpf.writerow([prob_pred[i], yp, ya, data[i][0]])
				if len(output_file_paths)>1 and yp != ya:
					misf.writerow([i+2, prob_pred[i], yp, ya, data[i][0]])

	return test_accuracy, avg_fscore


"""
Function for cross-validation
"""
def cv(configs, data_whole, labels, layers, static_idx, nonstatic_idx, info_file_path):
	stamp = str(datetime.now()).replace(' ','_').replace(':','-')
	model_file_path = 'cvmodel_' + stamp + '.p'
	nonstatic_file_path = 'cvwordvecs_nonstatic_' + stamp + '.p'
	static_file_path = 'cvwordvecs_static_' + stamp + '.p'
	
	avg_cv_accuracy = []
	avg_cv_fscore = []
	for repeat in range(configs['cv_repeats']):
		print_status('\n\nCROSS-VALIDATION REPEAT ' + str(repeat+1) + ' ::', info_file_path)
		
		shuffle(data_whole)		
		size = np.array_split(range(len(data_whole)),configs['cv_folds'])
		data = [data_whole[s[0]:s[-1]+1] for s in size]
		vocab = []		
		for fold in range(configs['cv_folds']):
			vocab.append (get_vocab(data[fold]))

		for fold in range(configs['cv_folds']):
			print_status('\n\tCROSS-VALIDATION Fold ' + str(fold+1) + ' : Training', info_file_path)
			training(configs=configs, existing_model_file='', existing_nonstatic_file='', validation_data=[], layers=layers, static_idx=static_idx, nonstatic_idx=nonstatic_idx, data=sum([data[k] for k in range(configs['cv_folds']) if k!=fold],[]), labels=labels, model_file_path=model_file_path, info_file_path=info_file_path, nonstatic_file_path=nonstatic_file_path, static_file_path=static_file_path)
				
			print_status('\n\tCV Test Fold ' + str(fold+1) + ' : Testing', info_file_path)
			test_accuracy, avg_fscore = testing(data=data[fold], threshold=-1, word_vec_files=configs['wordvec_files'], model_file_path=model_file_path, nonstatic_file_path=nonstatic_file_path, static_file_path=static_file_path, info_file_path=info_file_path)

			avg_cv_accuracy.append(test_accuracy)
			avg_cv_fscore.append(avg_fscore)

	os.remove(model_file_path)
	if nonstatic_idx>=0:
		os.remove(nonstatic_file_path)
	if static_idx>=0:
		os.remove(static_file_path)

	avg_cv_fscore = np.asarray(avg_cv_fscore).mean()
	avg_cv_accuracy = np.asarray(avg_cv_accuracy).mean()
	print_status('\n\n************************************************************************************', info_file_path)
	print_status('OVERALL CROSS-VALIDATION AVERAGE ACCURACY : ' + str(round(avg_cv_accuracy,3)) + '%', info_file_path)
	print_status('OVERALL CROSS-VALIDATION AVERAGE FSCORE   : ' + str(round(avg_cv_fscore,3)) + '%', info_file_path)
	print_status('************************************************************************************\n\n', info_file_path)
	

"""
Function to classify raw text
"""
def classify(text, preprocess, threshold, Word_idx_map_static, Word_idx_map_nonstatic, test_model):
	for p in preprocess:
		text = p(text)
	words = text.split(' ')
	string = []
	
	if Word_idx_map_static!=None:
		Xtest_static = []
		for wd in words:
			if wd in Word_idx_map_static:
				Xtest_static.append(Word_idx_map_static[wd])
				string.append(wd)

	if Word_idx_map_nonstatic!=None:
		Xtest_nonstatic = []
		string = []
		for wd in words:
			if wd in Word_idx_map_nonstatic:
				Xtest_nonstatic.append(Word_idx_map_nonstatic[wd])
				string.append(wd)

	if Word_idx_map_static!=None and Word_idx_map_nonstatic!=None:
		prob_pred = test_model(np.asarray(Xtest_static, dtype='int32'), np.asarray(Xtest_nonstatic, dtype='int32'))
	elif Word_idx_map_static!=None:
		prob_pred = test_model(np.asarray(Xtest_static, dtype='int32'))
	elif Word_idx_map_nonstatic!=None:
		prob_pred = test_model(np.asarray(Xtest_nonstatic, dtype='int32'))
	
	if threshold == 1:
		y_pred = prob_pred.argmax()
	elif threshold == 0:
		y_pred = prob_pred.argmin()
	else:
		y_pred = prob_pred >= threshold
		y_pred = y_pred.astype(int)

	# returns processed string, predicted probabilities and predicted outputs
	return [' '.join(string), prob_pred, y_pred]

