import tensorflow as tf
import sklearn
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import numpy
import random
import pandas as pd

d = pd.read_csv('santander/train.csv')

X_train_list=[]
Y_train_list=[]
X_validate_list=[]
Y_validate_list=[]
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
	X_train_list.append(innerlist[:369])
	if innerlist[370]==0:
		Y_train_list.append([1,0])
	else:
		Y_train_list.append([0,1])

for innerlist in validate_data:
	X_validate_list.append(innerlist[:369])
	if innerlist[370]==0:
		Y_validate_list.append([1,0])
	else:
		Y_validate_list.append([0,1])

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train_list)
X_validate = scaler.fit_transform(X_validate_list)
print ("Scaled")
X_train_list=[]
X_validate_list=[]
X_train_list_2 = X_train.tolist()
X_validate_list_2 = X_validate.tolist()
x=tf.placeholder(tf.float32, [None, 369])
X_train=[]
X_validate=[]
data_list=[]
d=[]
training_data=[]
validate_data=[]

#W1 = tf.Variable(tf.zeros([13,5]))
#W2 = tf.Variable(tf.zeros([5,1]))
#W = tf.Variable(tf.random_normal([13,1], mean = 0.0,stddev=0.35))
#b1 = tf.Variable(tf.zeros([1]))
#b2 = tf.Variable(tf.zeros([1]))

W = tf.Variable(tf.zeros([369,2]))
b = tf.Variable(tf.zeros([2]))
#y = tf.nn.softmax(tf.matmul(x,W) + b)
#h1 = tf.nn.relu(tf.matmul(x,W1) + b1)
#y = tf.matmul(h1,W2) + b2
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#sse = tf.reduce_sum( tf.pow((y_ - y),2))
sse = tf.reduce_sum( (y_ - y)*(y_ - y))
mse = tf.reduce_mean( tf.pow((y_ - y),2))

train_step = tf.train.GradientDescentOptimizer(0.000000000000000001).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
correct_predition = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_predition, "float"))

index_list = range(len(X_train_list_2))
for i in range(50):
	#1000000 3489.25
	#batch_xs, batch_ys = shuffle(X_train_list, Y_train_list, random_state=i)
	#for (x,y) in zip(batch_xs, batch_ys):
		#sess.run(train_step, feed_dict={x: x, y_: y})
	
	random.shuffle(index_list)
	tempx=[]
	tempy=[]
	##print index_list

	count=0
	for j in index_list:
		if count > 500:
			break
		tempx.append(X_train_list_2[j])
		tempy.append(Y_train_list[j])
	sess.run(train_step, feed_dict={x: tempx, y_: tempy})
	if i%10 == 0:
		print sess.run(accuracy, feed_dict={x: tempx, 	y_: tempy})


	#print sess.run(sse, feed_dict={x: tempx, y_: tempy})
	#sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	#sess.run(train_step, feed_dict={x: X_train[i], y_: Y_train2[i]})
print "Done training"
#print sess.run(sse, feed_dict={x: X_test, y_: Y_test2})
print "W:"
print sess.run(W)
print "b:"
print sess.run(b)
print sess.run(accuracy, feed_dict={x: X_validate_list_2, y_: Y_validate_list})
print "y:"
print sess.run(y, feed_dict={x: X_validate_list_2})
print "y_:"
print sess.run(y_, feed_dict={y_: Y_validate_list})
#print sess.run(y, feed_dict={x: X_validate_list})
#print Y_validate_list
#test= pd.read_csv('santander/test.csv')
#print sess.run(y, feed_dict={x: X_validate_list})
