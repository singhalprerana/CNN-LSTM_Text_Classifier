"""
CODE FOR NEURAL NETWORK LAYERS FOR CLASSIFICATION
Some of the code is modified from
- https://github.com/yoonkim/CNN_sentence (Convolutional Neural Networks for Sentence Classification)
- deeplearning.net/tutorial (for ConvNet and LSTM classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
__author__ = "Prerana Singhal"

import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from collections import defaultdict, OrderedDict



"""
Activation functions
"""
def ReLU(x):
	y = T.maximum(0.0, x)
	return(y)
def Sigmoid(x):
	y = T.nnet.sigmoid(x)
	return(y)
def Softmax(x):
	y = T.nnet.softmax(x)
	return(y)
def Tanh(x):
	y = T.tanh(x)
	return(y)
def Iden(x):
	y = x
	return(y)


"""
Dropout function
"""
def dropout_from_layer(rng, layer, p):     
	"""
	Function for applying dropout to a layer (p is the probablity of dropping a unit)
	"""
	srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
	# p=1-p because 1's indicate keep and p is prob of dropping
	mask = srng.binomial(n=1, p=1-p, size=layer.shape)
	# The cast is important because
	# int * float32 = float64 which pulls things off the gpu
	output = layer * T.cast(mask, theano.config.floatX)
	#output = layer * numpy.float32(numpy.random.binomial([numpy.ones(layer.shape,dtype=theano.config.floatX)],1-p))
	return output


"""
Error functions
"""
def square_error(output, act_y):
	"""
	Function to return the squared error value of the layer
	output : a vector :: predicted values of output layer neurons
	act_y : a vector :: actual labels (expected values of output layer neurons) for the input
	"""
	return T.sum((output - act_y) ** 2)

def negative_log_likelihood(output, act_y):
	"""
	Function to return the negative log-likelihood value of Layer
	act_y : a vector :: actual labels (expected values of output layer neurons) for the input
	"""

	return -T.mean((act_y * T.log(output + 0.000001)) + ((1 - act_y) * T.log(1 - output + 0.000001)))


"""
Back-Propagation Update functions
"""
def sgd_updates_adadelta(params, cost, rho):
	"""
	adadelta update rule, mostly from
	https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
	"""
	def as_floatX(variable):
		if isinstance(variable, float):
			return numpy.cast[theano.config.floatX](variable)

		if isinstance(variable, numpy.ndarray):
			return numpy.cast[theano.config.floatX](variable)
		return theano.tensor.cast(variable, theano.config.floatX)

	input_name='NON_STATIC_INPUT'
	epsilon=1e-6
	norm_lim=9,
	updates = OrderedDict({})
	exp_sqr_grads = OrderedDict({})
	exp_sqr_ups = OrderedDict({})
	gparams = []
	for param in params:
		empty = numpy.zeros_like(param.get_value())
		exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
		gp = T.grad(cost, param)
		exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
		gparams.append(gp)
	for param, gp in zip(params, gparams):
		exp_sg = exp_sqr_grads[param]
		exp_su = exp_sqr_ups[param]
		up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
		updates[exp_sg] = T.cast(up_exp_sg, dtype=theano.config.floatX)
		step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
		updates[exp_su] = T.cast(rho * exp_su + (1 - rho) * T.sqr(step), dtype=theano.config.floatX)
		stepped_param = param + step
		if (param.get_value(borrow=True).ndim == 2) and (param.name!=input_name):
			col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
			desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
			scale = desired_norms / (1e-7 + col_norms)
			updates[param] = T.cast(stepped_param * scale, dtype=theano.config.floatX)
		else:
			updates[param] = T.cast(stepped_param, dtype=theano.config.floatX)
	return updates

def updates_gradient(params, cost, rho):
	"""
	Simple update using gradient and rho = learning rate
	"""
	updates = []
	for param in params:
		dparam = T.grad(cost, param)
		updates.append((param, param - rho * dparam))
	return updates


"""
Pooling (down-sampling) functions
"""
def Mean_pooling(inp):
	"""
	Finding mean across rows; inp is a 2D matrix
	"""
	if inp.ndim==1:
		return T.mean(inp)
	else:
		return T.mean(inp,axis=0)

def Max_pooling(inp):
	"""
	Finding max across rows; inp is a 2D matrix
	"""
	if inp.ndim==1:
		return T.max(inp)
	else:
		return T.max(inp,axis=0)



"""
Class for a Fully Connected Layer
"""
class FullyConnectedLayer(object):
	
	def __init__(self, rng, n_in, n_out, activation, pooling, W, b, use_bias):
		self.activation = activation
		self.use_bias = use_bias
		self.pooling = pooling
		self.input = None

		if W is None:
			if activation.func_name == "ReLU":
				W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
			else:
				W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
													 size=(n_in, n_out)), dtype=theano.config.floatX)
			W = theano.shared(value=W_values, name='W')
		self.W = W

		# parameters of the model (bias or no bias)
		if use_bias:
			if b is None:
				b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
				b = theano.shared(value=b_values, name='b')
			self.b = b
			self.params = [self.W, self.b]
		else:
			self.params = [self.W]


	def predict(self, input):   #input is a vector (1D np.array)
		self.input = input
		if self.use_bias:
			output = T.dot(input, self.W) + self.b
		else:
			output = T.dot(input, self.W)

		self.output = (output if self.activation is None else self.activation(output))
		if self.pooling != None:
			self.output = self.pooling(self.output)
		return self.output

	def predict_dropout(self, input, rng, p):   #input is a vector (1D np.array)
		if self.input != input:
			self.input = input
			self.output = self.predict(input)

		self.dropout_output = dropout_from_layer(rng, self.output, p)
		return self.dropout_output


	def __getstate__(self):
		if self.use_bias:
			return (self.activation,self.use_bias,self.pooling,self.W.get_value(),self.b.get_value())
		else:
			return (self.activation,self.use_bias,self.pooling,self.W.get_value())

	def __setstate__(self, state):
		self.activation = state[0]
		self.use_bias = state[1]
		self.pooling = state[2]
		self.W = theano.shared(value=numpy.asarray(state[3], dtype=theano.config.floatX), name='W')
		if self.use_bias:
			self.b = theano.shared(value=numpy.asarray(state[4], dtype=theano.config.floatX), name='b')
			self.params = [self.W, self.b]
		else:
			self.params = [self.W]



"""
Class for a Convolution Layer
"""
class ConvolutionLayer(object):
	
	def __init__(self, rng, dim_in, dim_out, window, activation, pooling, W, b, use_bias):
		self.activation = activation
		self.use_bias = use_bias
		self.pooling = pooling
		self.window = window
		self.dim_in = dim_in
		self.input = None

		n_in = dim_in * window
		n_out = dim_out

		if W is None:
			if activation.func_name == "ReLU":
				W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
			else:
				W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
													 size=(n_in, n_out)), dtype=theano.config.floatX)
			W = theano.shared(value=W_values, name='W')
		self.W = W

		# parameters of the model (bias or no bias)
		if use_bias:
			if b is None:
				b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
				b = theano.shared(value=b_values, name='b')
			self.b = b
			self.params = [self.W, self.b]
		else:
			self.params = [self.W]


	def predict(self, input):   #input is a vector (1D np.array)
		self.input = input
		padding = numpy.asarray([numpy.zeros((self.dim_in,), dtype=theano.config.floatX)] * (self.window))
		inp = T.concatenate((padding, input, padding), axis=0)
		seq = T.arange(T.shape(inp)[0] - self.window + 1)
		self.input, _ = theano.scan(lambda v: inp[v : v+self.window].flatten(), sequences=seq)

		if self.use_bias:
			output = T.dot(self.input, self.W) + self.b
		else:
			output = T.dot(self.input, self.W)

		self.output = (output if self.activation is None else self.activation(output))
		if self.pooling != None:
			self.output = self.pooling(self.output)
		return self.output

	def predict_dropout(self, input, rng, p):   #input is a vector (1D np.array)
		if self.input != input:
			self.input = input
			self.output = self.predict(input)
		
		if self.pooling!=None:
			self.dropout_output = dropout_from_layer(rng, self.output, p)
		else:
			self.dropout_output = self.output

		return self.dropout_output


	def __getstate__(self):
		if self.use_bias:
			return (self.activation,self.window,self.dim_in,self.use_bias,self.pooling,self.W.get_value(),self.b.get_value())
		else:
			return (self.activation,self.window,self.dim_in,self.use_bias,self.pooling,self.W.get_value())

	def __setstate__(self, state):
		self.activation = state[0]
		self.window = state[1]
		self.dim_in = state[2]
		self.use_bias = state[3]
		self.pooling = state[4]
		self.W = theano.shared(value=numpy.asarray(state[5], dtype=theano.config.floatX), name='W')
		if self.use_bias:
			self.b = theano.shared(value=numpy.asarray(state[6], dtype=theano.config.floatX), name='b')
			self.params = [self.W, self.b]
		else:
			self.params = [self.W]



"""
Class for a Long Short Term Memory Layer
"""
class LSTMLayer(object):
	
	def __init__(self, rng, dim_in, dim_out, pooling, window, Wx, Wh, b, use_bias, use_last_output):
		self.use_bias = use_bias
		self.use_last_output = use_last_output
		self.window = window
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.pooling = pooling
		self.input = None
		gates = ['i', 'f', 'o', 'c']    # input, forget, output and memory gates

		# Weights for input-to-gate computations
		n_in = dim_in * window
		n_out = dim_out
		for i in range(4):
			if Wx[i] is None:
				W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
														 size=(n_in, n_out)), dtype=theano.config.floatX)
				Wx[i] = theano.shared(value=W_values, name='Wx'+gates[i])
		self.Wxi = Wx[0]
		self.Wxf = Wx[1]
		self.Wxo = Wx[2]
		self.Wxc = Wx[3]

		# Weights for hidden-to-gate computations
		n_in = dim_out
		n_out = dim_out
		for i in range(4):
			if Wh[i] is None:
				W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
														 size=(n_in, n_out)), dtype=theano.config.floatX)
				Wh[i] = theano.shared(value=W_values, name='Wh'+gates[i])
		self.Whi = Wh[0]
		self.Whf = Wh[1]
		self.Who = Wh[2]
		self.Whc = Wh[3]

		# parameters of the model (bias or no bias)
		if use_bias:
			for i in range(4):
				if b[i] is None:
					b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
					b[i] = theano.shared(value=b_values, name='b'+gates[i])
			self.bi = b[0]
			self.bf = b[1]
			self.bo = b[2]
			self.bc = b[3]
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc, self.bi, self.bf, self.bo, self.bc]
		else:
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc]

	
	def predict(self, input):   #input is an array of vectors (2D np.array)
		self.input = input
		padding = numpy.asarray([numpy.zeros((self.dim_in,), dtype=theano.config.floatX)] * (self.window))
		inp = T.concatenate((padding, input, padding), axis=0)
		seq = T.arange(T.shape(inp)[0] - self.window + 1)
		self.input, _ = theano.scan(lambda v: inp[v : v+self.window].flatten(), sequences=seq)

		# initialize the gates
		cgate = theano.shared(numpy.zeros((self.dim_out,), dtype=theano.config.floatX))
		hidden = T.tanh(cgate)

		# gate computations
		def lstm_step(x, h_prev, c_prev):
			if self.use_bias:
				igate = T.nnet.sigmoid(T.dot(x, self.Wxi) + T.dot(h_prev, self.Whi) + self.bi)
			else:
				igate = T.nnet.sigmoid(T.dot(x, self.Wxi) + T.dot(h_prev, self.Whi))
			if self.use_bias:
				fgate = T.nnet.sigmoid(T.dot(x, self.Wxf) + T.dot(h_prev, self.Whf) + self.bf)
			else:
				fgate = T.nnet.sigmoid(T.dot(x, self.Wxf) + T.dot(h_prev, self.Whf))
			if self.use_bias:
				ogate = T.nnet.sigmoid(T.dot(x, self.Wxo) + T.dot(h_prev, self.Who) + self.bo)
			else:
				ogate = T.nnet.sigmoid(T.dot(x, self.Wxo) + T.dot(h_prev, self.Who))
			if self.use_bias:
				cgate = (fgate * c_prev) + (igate * T.tanh(T.dot(x, self.Wxc) + T.dot(h_prev, self.Whc) + self.bc))
			else:
				cgate = (fgate * c_prev) + (igate * T.tanh(T.dot(x, self.Wxc) + T.dot(h_prev, self.Whc)))
			hidden = (ogate * T.tanh(cgate))
			return hidden, cgate

		[self.output, _], _ = theano.scan(fn=lstm_step, 
								  sequences = dict(input=self.input, taps=[0]), 
								  outputs_info = [hidden, cgate])
		if self.use_last_output:
			self.output = self.output[-1]
		if self.pooling != None:
			self.output = self.pooling(self.output)
		return self.output

	def predict_dropout(self, input, rng, p):   #input is a vector (1D np.array)
		if self.input != input:
			self.input = input
			self.output = self.predict(input)

		if self.pooling!=None:
			self.dropout_output = dropout_from_layer(rng, self.output, p)
		else:
			self.dropout_output = self.output
		return self.dropout_output


	def __getstate__(self):
		if self.use_bias:
			return (self.use_bias,self.use_last_output,self.pooling,self.window,self.dim_in,self.dim_out,self.Wxi.get_value(),self.Wxf.get_value(),self.Wxo.get_value(),self.Wxc.get_value(),self.Whi.get_value(),self.Whf.get_value(),self.Who.get_value(),self.Whc.get_value(),self.bi.get_value(),self.bf.get_value(),self.bo.get_value(),self.bc.get_value())
		else:
			return (self.use_bias,self.use_last_output,self.pooling,self.window,self.dim_in,self.dim_out,self.Wxi.get_value(),self.Wxf.get_value(),self.Wxo.get_value(),self.Wxc.get_value(),self.Whi.get_value(),self.Whf.get_value(),self.Who.get_value(),self.Whc.get_value())

	def __setstate__(self, state):
		self.use_bias = state[0]
		self.use_last_output = state[1]
		self.pooling = state[2]
		self.window = state[3]
		self.dim_in = state[4]
		self.dim_out = state[5]
		self.Wxi = theano.shared(value=numpy.asarray(state[6], dtype=theano.config.floatX), name='Wxi')
		self.Wxf = theano.shared(value=numpy.asarray(state[7], dtype=theano.config.floatX), name='Wxf')
		self.Wxo = theano.shared(value=numpy.asarray(state[8], dtype=theano.config.floatX), name='Wxo')
		self.Wxc = theano.shared(value=numpy.asarray(state[9], dtype=theano.config.floatX), name='Wxc')
		self.Whi = theano.shared(value=numpy.asarray(state[10], dtype=theano.config.floatX), name='Whi')
		self.Whf = theano.shared(value=numpy.asarray(state[11], dtype=theano.config.floatX), name='Whf')
		self.Who = theano.shared(value=numpy.asarray(state[12], dtype=theano.config.floatX), name='Who')
		self.Whc = theano.shared(value=numpy.asarray(state[13], dtype=theano.config.floatX), name='Whc')
		if self.use_bias:
			self.bi = theano.shared(value=numpy.asarray(state[14], dtype=theano.config.floatX), name='bi')
			self.bf = theano.shared(value=numpy.asarray(state[15], dtype=theano.config.floatX), name='bf')
			self.bo = theano.shared(value=numpy.asarray(state[16], dtype=theano.config.floatX), name='bo')
			self.bc = theano.shared(value=numpy.asarray(state[17], dtype=theano.config.floatX), name='bc')
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc, self.bi, self.bf, self.bo, self.bc]
		else:
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc]



"""
Class for a Long Short Term Memory Layer
"""
class ModifiedLSTMLayer(object):
	
	def __init__(self, rng, dim_in, dim_out, pooling, window, Wx, Wh, b, use_bias, use_last_output):
		self.use_bias = use_bias
		self.use_last_output = use_last_output
		self.window = window
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.pooling = pooling
		self.input = None
		gates = ['i', 'f', 'o', 'c']    # input, forget, output and memory gates

		# Weights for input-to-gate computations
		n_in = dim_in * window
		n_out = dim_out
		for i in range(4):
			if Wx[i] is None:
				W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
														 size=(n_in, n_out)), dtype=theano.config.floatX)
				Wx[i] = theano.shared(value=W_values, name='Wx'+gates[i])
		self.Wxi = Wx[0]
		self.Wxf = Wx[1]
		self.Wxo = Wx[2]
		self.Wxc = Wx[3]

		# Weights for hidden-to-gate computations
		n_in = dim_out
		n_out = dim_out
		for i in range(4):
			if Wh[i] is None:
				W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
														 size=(n_in, n_out)), dtype=theano.config.floatX)
				Wh[i] = theano.shared(value=W_values, name='Wh'+gates[i])
		self.Whi = Wh[0]
		self.Whf = Wh[1]
		self.Who = Wh[2]
		self.Whc = Wh[3]

		# parameters of the model (bias or no bias)
		if use_bias:
			for i in range(4):
				if b[i] is None:
					b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
					b[i] = theano.shared(value=b_values, name='b'+gates[i])
			self.bi = b[0]
			self.bf = b[1]
			self.bo = b[2]
			self.bc = b[3]
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc, self.bi, self.bf, self.bo, self.bc]
		else:
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc]

	
	def predict(self, input):   #input is an array of vectors (2D np.array)
		self.input = input
		padding = numpy.asarray([numpy.zeros((self.dim_in,), dtype=theano.config.floatX)] * (self.window))
		inp = T.concatenate((padding, input, padding), axis=0)
		seq = T.arange(T.shape(inp)[0] - self.window + 1)
		self.input, _ = theano.scan(lambda v: inp[v : v+self.window].flatten(), sequences=seq)

		# initialize the gates
		cgate = theano.shared(numpy.zeros((self.dim_out,), dtype=theano.config.floatX))
		hidden = T.tanh(cgate)

		# gate computations
		def lstm_step(x, h_prev, c_prev):
			if self.use_bias:
				igate = T.nnet.sigmoid(T.dot(x, self.Wxi) + T.dot(h_prev, self.Whi) + self.bi)
			else:
				igate = T.nnet.sigmoid(T.dot(x, self.Wxi) + T.dot(h_prev, self.Whi))
			if self.use_bias:
				fgate = T.nnet.sigmoid(T.dot(x, self.Wxf) + T.dot(h_prev, self.Whf) + self.bf)
			else:
				fgate = T.nnet.sigmoid(T.dot(x, self.Wxf) + T.dot(h_prev, self.Whf))
			if self.use_bias:
				ogate = T.nnet.sigmoid(T.dot(x, self.Wxo) + T.dot(h_prev, self.Who) + self.bo)
			else:
				ogate = T.nnet.sigmoid(T.dot(x, self.Wxo) + T.dot(h_prev, self.Who))
			if self.use_bias:
				cgate = (fgate * c_prev) + (igate * T.tanh(T.dot(x, self.Wxc) + T.dot(h_prev, self.Whc) + self.bc))
			else:
				cgate = (fgate * c_prev) + (igate * T.tanh(T.dot(x, self.Wxc) + T.dot(h_prev, self.Whc)))
			hidden = (ogate * T.tanh(cgate))
			return hidden, cgate

		[self.output, _], _ = theano.scan(fn=lstm_step, 
								  sequences = dict(input=self.input, taps=[0]), 
								  outputs_info = [hidden, cgate])
		if self.use_last_output:
			self.output = self.output[-1]
		if self.pooling != None:
			self.output = self.pooling(self.output)
		return self.output

	def predict_dropout(self, input, rng, p):   #input is a vector (1D np.array)
		if self.input != input:
			self.input = input
			self.output = self.predict(input)

		if self.pooling!=None:
			self.dropout_output = dropout_from_layer(rng, self.output, p)
		else:
			self.dropout_output = self.output
		return self.dropout_output


	def __getstate__(self):
		if self.use_bias:
			return (self.use_bias,self.use_last_output,self.pooling,self.window,self.dim_in,self.dim_out,self.Wxi.get_value(),self.Wxf.get_value(),self.Wxo.get_value(),self.Wxc.get_value(),self.Whi.get_value(),self.Whf.get_value(),self.Who.get_value(),self.Whc.get_value(),self.bi.get_value(),self.bf.get_value(),self.bo.get_value(),self.bc.get_value())
		else:
			return (self.use_bias,self.use_last_output,self.pooling,self.window,self.dim_in,self.dim_out,self.Wxi.get_value(),self.Wxf.get_value(),self.Wxo.get_value(),self.Wxc.get_value(),self.Whi.get_value(),self.Whf.get_value(),self.Who.get_value(),self.Whc.get_value())

	def __setstate__(self, state):
		self.use_bias = state[0]
		self.use_last_output = state[1]
		self.pooling = state[2]
		self.window = state[3]
		self.dim_in = state[4]
		self.dim_out = state[5]
		self.Wxi = theano.shared(value=numpy.asarray(state[6], dtype=theano.config.floatX), name='Wxi')
		self.Wxf = theano.shared(value=numpy.asarray(state[7], dtype=theano.config.floatX), name='Wxf')
		self.Wxo = theano.shared(value=numpy.asarray(state[8], dtype=theano.config.floatX), name='Wxo')
		self.Wxc = theano.shared(value=numpy.asarray(state[9], dtype=theano.config.floatX), name='Wxc')
		self.Whi = theano.shared(value=numpy.asarray(state[10], dtype=theano.config.floatX), name='Whi')
		self.Whf = theano.shared(value=numpy.asarray(state[11], dtype=theano.config.floatX), name='Whf')
		self.Who = theano.shared(value=numpy.asarray(state[12], dtype=theano.config.floatX), name='Who')
		self.Whc = theano.shared(value=numpy.asarray(state[13], dtype=theano.config.floatX), name='Whc')
		if self.use_bias:
			self.bi = theano.shared(value=numpy.asarray(state[14], dtype=theano.config.floatX), name='bi')
			self.bf = theano.shared(value=numpy.asarray(state[15], dtype=theano.config.floatX), name='bf')
			self.bo = theano.shared(value=numpy.asarray(state[16], dtype=theano.config.floatX), name='bo')
			self.bc = theano.shared(value=numpy.asarray(state[17], dtype=theano.config.floatX), name='bc')
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc, self.bi, self.bf, self.bo, self.bc]
		else:
			self.params = [self.Wxi, self.Wxf, self.Wxo, self.Wxc, self.Whi, self.Whf, self.Who, self.Whc]




