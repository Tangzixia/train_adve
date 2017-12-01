import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


reader=pywrap_tensorflow.NewCheckpointReader("H:\\train-adversarial\\inception_v3.ckpt")
var_to_shape_map=reader.get_variable_to_shape_map()

def all_var_list():
	result=[]
	for key in var_to_shape_map:
		result.append(key)
	return result
	#print ("tensor_name: ",key)
	#print (reader.get_tensor(key))
	#if(key=='vgg_16/fc7/biases'):
		#print (reader.get_tensor(key))
	#result.append(key)
