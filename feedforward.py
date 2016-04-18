import tensorflow as tf
import sklearn
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import numpy
import random
import pandas as pd
import math

d = pd.read_csv('santander/train.csv')

X_train_list=[]
Y_train_list=[]
X_validate_list=[]
Y_validate_list=[]

X_train_0_list=[]
X_train_1_list=[]
X_validate_0_list=[]
X_validate_1_list=[]
Y_train_0_list=[]
Y_train_1_list=[]
Y_validate_0_list=[]
Y_validate_1_list=[]

Ws=[]
Bs=[]

data_list=[]
for row in d.itertuples():
	#X_train_list.append(list(row[1:371]))
	#Y_train_list.append(list(row[372]))
	data_list.append(list(row[1:]))


random.shuffle(data_list)
datasize=len(data_list)
training_data = data_list[:datasize*60/100]
validate_data = data_list[datasize*60/100:]

for innerlist in training_data:
	#print len(innerlist)
	#X_train_list.append(innerlist[:369])
	if innerlist[370]==0:
		X_train_0_list.append(innerlist[:369])
		#Y_train_list.append([1,0])
		Y_train_0_list.append([1,0])
	else:
		X_train_1_list.append(innerlist[:369])
		#Y_train_list.append([0,1])
		Y_train_1_list.append([0,1])

for innerlist in validate_data:
	X_validate_list.append(innerlist[:369])
	if innerlist[370]==0:
		#X_train_0_list.append(innerlist[:369])
		Y_validate_list.append([1,0])
		#Y_validate_0_list.append([1,0])
	else:
		#X_train_1_list.append(innerlist[:369])
		Y_validate_list.append([0,1])
		#Y_validate_1_list.append([0,1])

#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train_list)
#X_validate = scaler.fit_transform(X_validate_list)
#print ("Scaled")
#X_train_list_2=X_train_list
#Y_train_list_2=Y_train_list
#X_train_list=[]
#X_validate_list=[]
#X_train_list_2 = X_train.tolist()
#X_validate_list_2 = X_validate.tolist()
x=tf.placeholder(tf.float32, [None, 369])
X_train=[]
X_validate=[]
data_list=[]
d=[]
training_data=[]
validate_data=[]
layer_sizes=[300,200,100,10,2]
next_layer_input=x
for dim in layer_sizes:
	input_dim = int(next_layer_input.get_shape()[1])

	# Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
	W = tf.Variable(tf.random_uniform([input_dim, dim],-1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
	#W = tf.Variable(tf.zeros([input_dim,dim]))
	# Initialize b to zero
	b = tf.Variable(tf.zeros([dim]))

	Ws.append(W)
	Bs.append(b)
	#output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
	if (dim==2):
		output = tf.nn.softmax(tf.matmul(next_layer_input,W) + b)	
		#tf.nn.softmax_cross_entropy_with_logits(tf.matmul(next_layer_input,W) + b)
	else:
		#output = tf.nn.relu(tf.matmul(next_layer_input,W) + b)
		output = tf.matmul(next_layer_input,W) + b
	# the input into the next layer is the output of this layer
	next_layer_input = output



#W = tf.Variable(tf.zeros([369,2]))
#b = tf.Variable(tf.zeros([2]))
#y = tf.nn.softmax(tf.matmul(x,W) + b)
y=next_layer_input
y_ = tf.placeholder(tf.float32, [None, 2])

#cross_entropy = -tf.reduce_sum(y_ * tf.log(y+1e-9))

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#sse = tf.reduce_sum( tf.pow((y_ - y),2))
sse = tf.reduce_sum( (y_ - y)*(y_ - y))
mse = tf.reduce_mean( tf.pow((y_ - y),2))

train_step = tf.train.GradientDescentOptimizer(0.00000000000000000000001).minimize(mse)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
correct_predition = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_predition, "float"))

index_0_list = range(len(X_train_0_list))
index_1_list = range(len(X_train_1_list))

for i in range(50):
	#1000000 3489.25
	#batch_xs, batch_ys = shuffle(X_train_list, Y_train_list, random_state=i)
	#for (x,y) in zip(batch_xs, batch_ys):
		#sess.run(train_step, feed_dict={x: x, y_: y})
	
	random.shuffle(index_0_list)
	random.shuffle(index_1_list)
	tempx=[]
	tempy=[]
	##print index_list

	count=0
	for j in index_0_list:
		if count > 1000:
			break
		tempx.append(X_train_0_list[j])
		tempy.append(Y_train_0_list[j])
		count+=1
	count=0
	for j in index_1_list:
		if count > 1000:
			break
		tempx.append(X_train_1_list[j])
		tempy.append(Y_train_1_list[j])
		count+=1
	sess.run(train_step, feed_dict={x: tempx, y_: tempy})
	if i%10 == 0:
		print sess.run(accuracy, feed_dict={x: tempx, 	y_: tempy})


	#print sess.run(sse, feed_dict={x: tempx, y_: tempy})
	#sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	#sess.run(train_step, feed_dict={x: X_train[i], y_: Y_train2[i]})
print "Done training"
#print sess.run(sse, feed_dict={x: X_test, y_: Y_test2})
#print "W:"
#print sess.run(Ws)
#print "b:"
#print sess.run(Bs)
#print sess.run(accuracy, feed_dict={x: X_validate_list, y_: Y_validate_list})
#print "y:"
#print sess.run(y, feed_dict={x: X_validate_list})
#print "y_:"
#print sess.run(y_, feed_dict={y_: Y_validate_list})
#print sess.run(y, feed_dict={x: X_validate_list})
#print Y_validate_list
test= pd.read_csv('santander/test.csv')
data_list=[]
for row in test.itertuples():
	#X_train_list.append(list(row[1:371]))
	#Y_train_list.append(list(row[372]))
	data_list.append(list(row[1:]))


prediction = tf.argmax(y,1)
X_test_list=[]
for innerlist in data_list:
	X_test_list.append(innerlist[:369])

print "y"
file = open("submission", 'w')
print >>file,sess.run(prediction, feed_dict={x: X_test_list})
