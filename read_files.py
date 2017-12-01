#coding=utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

def to_categorial(y,n_classes):
    y_std=np.zeros([len(y),n_classes])
    for i in range(len(y)):
        y_std[i,y[i]]=1.0
    return y_std

def file_label_map(path):
	map={}
	f=open(path,'r')
	line=f.readline()
	count=0;
	while(line):
		label=str(line).split(":")[0].split("[")[0]
		map[label]=count;
		line=f.readline()
		count=count+1
	f.close();
	return map
def getFileArr_std(dir,map_std,train=True):
	map_file_result={}
	map_file_label={}
	file_list=os.listdir(dir)
	for file in file_list:
		file_path=os.path.join(dir,file)
		label=file.split(".")[0].split("_")[0]
		print(label)
		img=Image.open(file_path).resize((299,299),Image.ANTIALIAS).convert("RGB")
		result=np.array([])
		r,g,b=img.split()
		r_arr=np.array(r).reshape(299*299)
		g_arr=np.array(g).reshape(299*299)
		b_arr=np.array(b).reshape(299*299)
		img_arr=np.concatenate((r_arr,g_arr,b_arr))
		result=np.concatenate((result,img_arr))
		result=result.reshape((299,299,3))
		map_file_result[file]=result
		map_file_label[file]=map_std[label]
	
	ret_arr=[]
	count_0=0
	for file in file_list:
		each_list=[]
		#112*112*3的一个三维数组，现在需要将其扩展为4维度，5954*112*112*3
		result=map_file_result[file]
		label=map_file_label[file]
		#print(label_one_zero)
		each_list.append(result)
		each_list.append(label)
		print(count_0)
		ret_arr.append(each_list)
		count_0=count_0+1
	if train:
		np.save('/media/common/helloworld/train-adversarial/train_adversarial_10.npy', ret_arr)
	else:
		np.save('/media/common/helloworld/train-adversarial/test_adversarial_10.npy', ret_arr)

def getFileArr(dir):
	result_arr=[]
	label_list=[]
	map={}
	map_file_result={}
	map_file_label={}
	map_new={}
	count_label=0
	count=0
	file_list=os.listdir(dir)
	for file in file_list:
		file_path=os.path.join(dir,file)
		label=file.split(".")[0].split("_")[0]
		img=Image.open(file_path)
		result=np.array([])
		if(len(img.split())!=3):
			map[file]="hahaha"
			#os.remove(file_path)
		else:
			map[file]=label
			if label not in label_list:
				label_list.append(label)
				map_new[label]=count_label
				count_label=count_label+1
			r,g,b=img.split()

			r_arr=np.array(r).reshape(299*299)
			g_arr=np.array(g).reshape(299*299)
			b_arr=np.array(b).reshape(299*299)
			img_arr=np.concatenate((r_arr,g_arr,b_arr))
			result=np.concatenate((result,img_arr))
			#result=result.reshape((3,112,112))
			result=result.reshape((299,299,3))
			map_file_result[file]=result
			result_arr.append(result)
			count=count+1

	for file in file_list:
		if map[file]!="hahaha":
			map_file_label[file]=map_new[map[file]]
		#map[file]=map_new[map[file]]
	
	ret_arr=[]
	for file in file_list:
		if map[file]!="hahaha":
			each_list=[]
			label_one_zero=np.zeros(count_label)
			#112*112*3的一个三维数组，现在需要将其扩展为4维度，5954*112*112*3
			result=map_file_result[file]
			label=map_file_label[file]
			#print(label_one_zero)
			each_list.append(result)
			each_list.append(label)
			ret_arr.append(each_list)
	np.save('F:\\train-adversarial\\train_adversarial.npy', ret_arr)
def load_data_std(train_dir,test_dir):
    train_data=np.load(train_dir)
    test_data=np.load(test_dir)
    X_train_non,y_train_non=train_data[:,0],train_data[:,1]
    X_test_non,y_test_non=test_data[:,0],test_data[:,1]

    X_train=np.zeros([len(X_train_non),299,299,3])
    X_test=np.zeros([len(X_test_non),299,299,3])
    for i in range(len(X_train_non)):
        X_train[i,:,:,:]=X_train_non[i]
    for i in range(len(X_test_non)):
        X_test[i,:,:,:]=X_test_non[i]
    #y_train_non=y_train_non.tolist()
    #y_test_non=y_test_non.tolist()
    y_train=to_categorial(y_train_non,1001)
    y_test=to_categorial(y_test_non,1001)   
    return (X_train,y_train),(X_test,y_test)

def load_data(test_dir):
    test_data=np.load(test_dir)
    X_test_non,y_test_non=test_data[:,0],test_data[:,1]
    X_test=np.zeros([len(X_test_non),299,299,3])
    for i in range(len(X_test_non)):
        X_test[i,:,:,:]=X_test_non[i]
    y_test=to_categorial(y_test_non,182)   
    return X_test,y_test
