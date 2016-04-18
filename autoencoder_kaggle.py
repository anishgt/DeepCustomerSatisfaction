""" Deep Auto-Encoder implementation
	
	An auto-encoder works as follows:

	Data of dimension k is reduced to a lower dimension j using a matrix multiplication:
	softmax(W*x + b)  = x'
	
	where W is matrix from R^k --> R^j

	A reconstruction matrix W' maps back from R^j --> R^k

	so our reconstruction function is softmax'(W' * x' + b') 

	Now the point of the auto-encoder is to create a reduction matrix (values for W, b) 
	that is "good" at reconstructing  the original data. 

	Thus we want to minimize  ||softmax'(W' * (softmax(W *x+ b)) + b')  - x||

	A deep auto-encoder is nothing more than stacking successive layers of these reductions.
"""
#code refrence https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5
import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split


features = ['var15', 'ind_var5', 'ind_var8_0', 'ind_var30', 'num_var5', 'num_var30', 'num_var42', 'var36', 'num_meses_var5_ult3']
def create(x, layer_sizes):

	# Build the encoding layers
	next_layer_input = x

	encoding_matrices = []
	for dim in layer_sizes:
		input_dim = int(next_layer_input.get_shape()[1])

		# Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
		W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
		#W = tf.Variable(tf.zeros([input_dim,dim]))
		# Initialize b to zero
		b = tf.Variable(tf.zeros([dim]))

		# We are going to use tied-weights so store the W matrix for later reference.
		encoding_matrices.append(W)

		#output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
		output = tf.matmul(next_layer_input,W) + b
		# the input into the next layer is the output of this layer
		next_layer_input = output

	# The fully encoded x value is now stored in the next_layer_input
	encoded_x = next_layer_input

	# build the reconstruction layers by reversing the reductions
	layer_sizes.reverse()
	encoding_matrices.reverse()


	for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
		# we are using tied weights, so just lookup the encoding matrix for this step and transpose it
		W = tf.transpose(encoding_matrices[i])
		b = tf.Variable(tf.zeros([dim]))
		output = tf.matmul(next_layer_input,W) + b
		next_layer_input = output

	# the fully encoded and reconstructed value of x is here:
	reconstructed_x = next_layer_input

	return {
		'encoded': encoded_x,
		'decoded': reconstructed_x,
		'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
	}

def error_thresh(x):
	return x>=29
	#return x>=0.45

def log_var(x):
	if x[7] == 0:
		#rep = 0.00000001
		rep=x[7]
	else:
		rep = x[7]
		
	return (x[0], x[1], x[2], x[3], x[4], x[5], x[6], rep, x[8], x[9])

def deep_test(data):
	sess = tf.Session()
	random.seed(1)
	train, test = train_test_split(data,test_size=0.1)
	print "test",test.shape
	print "train",train.shape
	data_filtered = train[train['TARGET']==0].ix[:,:9]
	test_pos = test[test['TARGET']==0]
	test_neg = test[test['TARGET']==1]
	print "no of filtered rows",data_filtered.shape
	total_pos = test_pos.shape[0]
	total_neg = test_neg.shape[0]
	print "no of positive rows",total_pos
	print "no of negative rows",total_neg	
	
	start_dim = data_filtered.shape[1]
	x = tf.placeholder("float", [None, start_dim])
	autoencoder = create(x, [9, 7, 5, 3, 2])
	# lyr = range(10,350)
	# lyr.reverse()
	# autoencoder = create(x,lyr)
	init = tf.initialize_all_variables()
	sess.run(init)
	train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(autoencoder['cost'])


	for i in range(65000):	
		data_filtered.reindex(np.random.permutation(data_filtered.index))
		batch = data_filtered.iloc[:,:].values[:100]
		sess.run(train_step, feed_dict={x: batch})
		#print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})

	#print i, " test cost_pos", sess.run(autoencoder['cost'], feed_dict={x:test_pos.iloc[:,:9].values})
	#print i, " test cost_neg", sess.run(autoencoder['cost'], feed_dict={x:test_neg.iloc[:,:9].values})
	test_pos_r = [sess.run(autoencoder['cost'], feed_dict={x: np.reshape(np.array(test_pos.ix[i,0:9]),(1,9))}) for i in test_pos.index]
	#test_pos_r = [sess.run(autoencoder['cost'], feed_dict={x: np.reshape(np.array(test_pos.ix[i,:]),(1,371))}) for i in test_pos.index]
	test_neg_r = [sess.run(autoencoder['cost'], feed_dict={x: np.reshape(np.array(test_neg.ix[i,0:9]),(1,9))}) for i in test_neg.index]
	
	test_kaggle=pd.read_csv('./santander/test.csv')
	result = pd.DataFrame()
	result['ID']=test_kaggle['ID']

	test_kaggle = test_kaggle[features]
	test_kaggle_r = [sess.run(autoencoder['cost'], feed_dict={x: np.reshape(np.array(test_kaggle.ix[i,0:9]),(1,9))}) for i in test_kaggle.index]
	
	test_labels = [1 if x>29 else 0 for x in test_kaggle_r]
	print 'sum of test labels',sum(test_labels)
	result['TARGET']=test_labels
	result.to_csv('submission.csv')
	pos_above_thresh= sum(1 for x in test_pos_r if error_thresh(x))
	neg_above_thresh= sum(1 for x in test_neg_r if error_thresh(x))
	
	print "% positives above threshold", (float(pos_above_thresh)/total_pos)*100 , "%"
	print "% negatives above threshold", (float(neg_above_thresh)/total_neg)*100 , "%"
	
	
	
	plt.plot(test_pos_r,'ro',color='b')
	plt.plot(test_neg_r,'ro',color='r')
	plt.show()
	# print "testing begins"
	# for i in test.index:	
	# 	print test.loc[i,'TARGET'],sess.run(autoencoder['cost'], feed_dict={x: np.reshape(np.array(test.ix[i,0:9]),(1,9))})

if __name__ == '__main__':
	train = pd.read_csv("./santander/train.csv")
	
	data = train[features+['TARGET']]
	#cols_to_norm = ['var15', 'num_var5', 'num_var30', 'num_var42', 'var36', 'num_meses_var5_ult3']
	#data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
	#data_trans = data.apply(log_var, axis=1)
	# data['var38mc'] = np.isclose(data.var38, 117310.979016)
	# data['logvar38'] = data.loc[~data['var38mc'], 'var38'].map(np.log)
	# data.loc[data['var38mc'], 'logvar38'] = 0
	# data.drop('var38', axis=1, inplace=True)
	deep_test(data)


