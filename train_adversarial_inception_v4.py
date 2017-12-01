#-*-coding=utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
#from tensorflow.contrib.slim.nets import inception_v3, inception_v3_arg_scope
from read_files import load_data
import numpy as np
import os
import Inception_v4
import random
#from Inception_v4 import inception_v4,inception_v4_arg_scope

height = 299
width = 299
channels = 3
num_classes=1001
num_epochs=300
#train_dir="/media/common/helloworld/train-adversarial/train_adversarial.npy"
#test_dir="/media/common/helloworld/train-adversarial/test_adversarial.npy"
train_dir="/media/common/helloworld/train-adversarial/train_adversarial_10.npy"
test_dir="/media/common/helloworld/train-adversarial/test_adversarial_10.npy"
def min(x,y):
	if x<=y:
		return x
	else:
		return y
if __name__=="__main__":
	current_epoch = tf.Variable(0)

	X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
	y = tf.placeholder(tf.float32,shape=[None,182])
	'''
	with slim.arg_scope(Inception_v4.inception_v4_arg_scope()):
		logits, end_points=Inception_v4.inception_v4(X,num_classes=num_classes,is_training=False)
	'''
	with slim.arg_scope(nets.inception.inception_v3_arg_scope()):
		logits, end_points=nets.inception.inception_v3(X,num_classes=num_classes,is_training=False)
	shape=logits.get_shape().as_list()
	dim=1
	for d in shape[1:]:
		dim*=d
	fc_=tf.reshape(logits,[-1,dim])
	
	fc0_weights=tf.get_variable(name="fc0_weights",shape=(1001,182),initializer=tf.contrib.layers.xavier_initializer())
	fc0_biases=tf.get_variable(name="fc0_biases",shape=(182),initializer=tf.contrib.layers.xavier_initializer())
	logits_=tf.nn.bias_add(tf.matmul(fc_,fc0_weights),fc0_biases)
	predictions=tf.nn.softmax(logits_)
	#cross_entropy = -tf.reduce_sum(y*tf.log(predictions))  
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_,labels=y))
	#cross_entropy_mean=tf.reduce_mean(cross_entropy)
	learning_rate = tf.train.exponential_decay(0.01,  
                                           current_epoch,  
                                           decay_steps=50,  
                                           decay_rate=0.03)
	'''
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=num_epochs)
	'''
	train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy,var_list=[fc0_weights,fc0_biases])
	correct_pred=tf.equal(tf.argmax(y,1),tf.argmax(predictions,1))
	#acc=tf.reduce_sum(tf.cast(correct_pred,tf.float32))
	sum_n=tf.reduce_sum(tf.cast(correct_pred,tf.float32))
	accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


	batches=[]
	x_train,y_train=load_data(train_dir)
	x_train=x_train.astype("float32")
	x_train/=255
	for i in range(len(x_train)):
		batches.append((x_train[i],y_train[i]))
	random.shuffle(batches)
	#y_train=y_train.astype("float32")
	x_test,y_test=load_data(test_dir)
	x_test=x_test.astype("float32")
	x_test/=255
	#y_test=y_test.astype("float32")
	
	print("嗯嗯呢")
	print(x_train.shape,y_train.shape)
	#with slim.arg_scope(nets.inception.inception_v3_arg_scope()):
	#create_graph(x_train,y_train)nan,
	#print (slim.get_model_variables())

	init_fn = slim.assign_from_checkpoint_fn(
		os.path.join("/media/common/helloworld/train-adversarial/", 'inception_v3.ckpt'),
		slim.get_model_variables())
	init_op=tf.global_variables_initializer()
	
	batch_te=20
	batch_size=20
	batch_list=[]
	batch_list_t=[]
	flag=False
	print("开始训练")
	'''
	#count代表批的数量，现在我们处理训练集
	if x_train.shape[0]%batch_size:
		count=int(x_train.shape[0]/batch_size)+1
		flag=True
	else:
		count=int(x_train.shape[0]/batch_size)
	for i in range(count):
		if(flag==True and i==count-1):
			batch_list.append((x_train[i*batch_te:min(i*batch_te+batch_te,x_train.shape[0])],y_train[i*batch_te:min(i*batch_te+batch_te,y_train.shape[0])]))
		else:
			batch_list.append((x_train[i*batch_te:i*batch_te+batch_te],y_train[i*batch_te:i*batch_te+batch_te]))
	
	#count_t代表测试集的批的数量，现在我们处理测试集
	if x_train.shape[0]%batch_size:
		count_t=int(x_test.shape[0]/batch_size)+1
		flag=True
	else:
		count_t=int(x_test.shape[0]/batch_size)
	for i in range(count_t):
		if(flag==True and i==count_t-1):
			batch_list_t.append((x_test[i*batch_te:min(i*batch_te+batch_te,x_test.shape[0])],y_test[i*batch_te:min(i*batch_te+batch_te,y_test.shape[0])]))
		else:
			batch_list_t.append((x_test[i*batch_te:i*batch_te+batch_te],y_test[i*batch_te:i*batch_te+batch_te]))
	'''
	# Execute graph
	with tf.Session() as sess:
		sess.run(init_op)
		init_fn(sess)
		print("哈哈")
		#for i in range(5):
			#for i in range(count):
				#sess.run(train_step,feed_dict={X:batch_list[i][0],y:batch_list[i][1]})
			#z=sess.run(cross_entropy,feed_dict={X:batch_list[0][0],y:batch_list[0][1]})
			#print(z)
		#print(z.shape)
		#print(np.max(z,1))
		#print(np.argmax(z,1))
		#print("求和:")
		#print(np.sum(z,1))
		acc_avg=0.0
		loss_avg=0.0
		#epoches=50
		for epoch in range(num_epochs):
			print("第"+str(epoch+1)+"次迭代的训练集合集和预测个数	损失	准确率：")
			for i in range(30):
				batch_list=[]
				j=0
				while j<100:
					data=random.choice(batches)
					batch_list.append((data[0],data[1]))
					j=j+1
				batch_list_train=[]
				batch_list_label=[]
				for z in range(len(batch_list)):
					batch_list_train.append(batch_list[i][0])
					batch_list_label.append(batch_list[i][1])
				ts=sess.run(train_step,feed_dict={X:batch_list_train,y:batch_list_label})
				sum_,loss,acc=sess.run([sum_n,cross_entropy,accuracy],feed_dict={X:batch_list_train,y:batch_list_label})
				loss_avg+=loss
				acc_avg+=acc
				print(sum_,loss,acc)
			print("第"+str(epoch+1)+"次迭代的测试集准确率：")
			'''
			acc_t=0
			for i in range(count_t):
				#print(str(i)+"+++++")
				acc_t=acc_t+sess.run(accuracy,feed_dict={X:batch_list_t[i][0],y:batch_list_t[i][1]})
			print(acc_t/count_t)
			'''
			print(sess.run(accuracy,feed_dict={X:x_test,y:y_test}))
		print("======================================================")
		print("迭代结束！")
		print(loss_avg/(30*num_epochs),acc_avg/(30*num_epochs))
			#print(sess.run(fc_,feed_dict={X:batch_list[i][0],y:batch_list[i][1]}))
			#sum_,accuracy_n,acc=sess.run([sum_n,cross_entropy,accuracy],feed_dict={X:x_test,y:y_test})
			#print(sum_)
			#print(accuracy_n)
			#print(np.argmax(y_test,1))
			#print("第"+str(epoch+1)+"次迭代的准确率:"+str(acc))
