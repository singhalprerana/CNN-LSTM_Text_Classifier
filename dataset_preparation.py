#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
CODE FOR DATASET PREPARATION FOR CLASSIFICATION
Some of the code is modified from
- https://github.com/yoonkim/CNN_sentence (Convolutional Neural Networks for Sentence Classification)
"""
__author__ = "Prerana Singhal"

import csv, re, string, cPickle
import numpy as np
import theano
import unicodedata
from collections import defaultdict, OrderedDict



"""
Preprocessing functions
"""
def to_lowercase(text):
	return text.lower().strip()

def remove_all_punctuations(text):
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	text = regex.sub(' ', text).strip()
	return " ".join(text.split())

def remove_basic_punctuations(text):
	text = text.replace('.','')
	text = text.replace(',','')
	text = text.replace('?','')
	text = text.replace('!','')
	text = text.replace(';','')
	text = text.replace('-',' ')
	return text

def remove_spaced_single_punctuations(text):
	wds = text.split()
	return " ".join([w for w in wds if len(w)>1 or re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', w).strip() != ''])

def space_out_punctuations(text):
	return re.sub(r"([\w\s]+|[^\w\s]+)\s*", r"\1 ", text)

def remove_numbers(text):
	return re.sub(r' \d+ ',' ', text)

def replace_numbers(text):
	return re.sub(r' \d+ ',' *#NUMBER#* ', text)

def replace_accents(text):
	text = text.decode('utf-8')
	text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
	text = text.replace('-LRB-','(')
	text = text.replace('-RRB-',')')
	return text


"""
Functions to extract data from file
"""
def extract_data(filenames, preprocess, delimiter, labels_present):
	"""
	filename : name of the csv file with NO headings/titles and
		first column as text (data-point) to be classified,
		subsequent column(s) (if any in case of labels_present = True) consist of the labels associated with this data-point
	preprocess : list of preprocessing functions to be applied to the text
	"""
	data = []   # data (input-output)
	if labels_present:
		labels = [] # classification labels
		for filename in filenames:
			with open(filename, "rb") as file:
				f=csv.reader(file, delimiter=delimiter)
				for line in f:
					data.append(line)
					for l in line[1:]:
						if l != '' and l not in labels:
							labels.append(l)
		labels.sort()
		num_labels = len(labels)
		num_data = len(data)
		for i in range(num_data):
			x = [0] * num_labels
			for l in data[i][1:]:
				x[labels.index(l)] = 1
			for p in preprocess:
				data[i][0] = p(data[i][0])
			data[i] = [data[i][0], np.asarray(x)]
		return data, labels
	else:
		for filename in filenames:
			with open(filename, "rb") as file:
				f=csv.reader(file, delimiter=delimiter)

				for line in f:
					for p in preprocess:
						line[0] = p(line[0])
					data.append([line[0]])
		return data

def get_vocab(data):
	"""
	Function to extract the vocabulary (words) from data
	"""
	vocab = []
	for line in data:
		for wrd in line[0].split(' '):
			vocab.append(wrd)
	return list(set(vocab)) # remove duplicates


"""
Function to obtain word embeddings from file
"""
def load_vecs(vocab, dim, filenames, add_unknown, variance_random, load_all=False):
	"""
	Loads dimx1 word vectors from file
	"""
	# in order of priority (in case of conflict, filenames[0] has higher priority over filenames[1] and so on)
	word_vecs = {}
	for i in range(len(filenames)-1,-1,-1): 
		fname = filenames[i]
		if fname=='':
			continue
		if ".bin" in fname:     # Google (Mikolov) word2vec (GoogleNews-vectors-negative300)
			vocabx = {vocab[i]:i for i in range(len(vocab))}
			with open(fname, "rb") as f:
				header = f.readline()
				vocab_size, layer1_size = map(int, header.split())
				binary_len = np.dtype('float32').itemsize * layer1_size
				i = 0
				for line in xrange(vocab_size):
					word = []
					while True:
						ch = f.read(1)
						if ch == ' ':
							word = ''.join(word)
							break
						if ch != '\n':
							word.append(ch)
					if load_all or word in vocabx:
						word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
					else:
						f.read(binary_len)
		else:
			if load_all:
				word_vecs.update(cPickle.load(open(fname,"rb")))    
			else:
				vecs = cPickle.load(open(fname,"rb"))   
				for word in vocab:
					if word in vecs:
						word_vecs[word] = vecs[word]
	
	if add_unknown:
		for wrd in vocab:
			if wrd not in word_vecs:    #randomly initialize word embeddings for new words
				word_vecs[wrd] = np.random.uniform(-variance_random,variance_random,dim) # variance_unknown = 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones

	return word_vecs


"""
Function to create word-to-vec mapping
"""
def get_word2vec_map(word_vecs, vocab, load_all=False):
	"""
	Function to get word-index-vector mapping
	Word_idx_map : dictionary for word to index mapping
	Word_idx_vecs : List containing index-wise word embeddings
	vocab : List of words to be considered
	dim : dimension of the word-vectors
	"""
	Word_idx_map = {}
	Word_idx_vecs = []
	
	if load_all:
		count=0
		for wrd in word_vecs:
			Word_idx_map[wrd] = count
			Word_idx_vecs.append(word_vecs[wrd])
			count+=1
	else:
		for wrd in vocab:
			if wrd not in Word_idx_map and wrd in word_vecs:
				Word_idx_map[wrd] = len(Word_idx_vecs)
				Word_idx_vecs.append(word_vecs[wrd])
	
	return Word_idx_map, Word_idx_vecs


"""
Function to transform data into proper format (indices to word vectors)
"""
def get_idx_from_sent(text, Word_idx_map):
	"""
	Transforms sentence into a list of indices.
	"""
	x = []
	words = text.split()
	for word in words:
		if word in Word_idx_map:
			x.append(int(Word_idx_map[word]))
	return x

def make_idx_data_cv(data, Word_idx_map, labels_present):
	"""
	Transforms sentences into a 2-d matrix.
	"""
	Xdata = np.asarray([np.asarray(get_idx_from_sent(row[0], Word_idx_map), dtype='int32') for row in data])
	if labels_present:
		Ydata = np.asarray([row[1] for row in data], dtype='int32')
		return Xdata, Ydata
	else:
		return Xdata