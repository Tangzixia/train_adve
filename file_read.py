#coding=utf-8
import os.path
import os
import numpy as np
import shutil
import read_files as rf
h_dir="/media/common/helloworld/train-adversarial/train-adversarial"
d_dir="/media/common/helloworld/train-adversarial/test-adversarial"
def load_data(dir=h_dir,dest_dir=d_dir):
	if os.path.exists(dest_dir)==False:
		os.mkdir(dest_dir)
	filelist=os.listdir(dir)
	all_file_list=[]
	for filename in filelist:
		file_class=filename.split(".")[0].split("_")[0]
		if file_class not in all_file_list:
			file_path=os.path.join(dir,filename)
			shutil.copy(file_path,dest_dir)
			os.remove(file_path)
			all_file_list.append(file_class)

def load_data_d_dir_2_png_dir(dir=h_dir,dest_dir=d_dir):
	if os.path.exists(dest_dir)==False:
		os.mkdir(dest_dir)
	filelist=os.listdir(dir)
	#all_file_list=[]
	for dir_file in filelist:
		filename_each=os.path.join(dir,dir_file)
		filename_each_list=os.listdir(filename_each)
		for filename in filename_each_list:
			print(filename)
			file_class=filename.split(".")[0].split("_")[0]
			count=0
			if count<5:
				file_path=os.path.join(filename_each,filename)
				print(file_path)
				shutil.copy(file_path,dest_dir)
				os.remove(file_path)
				#all_file_list.append(file_class)
				count=count+1
				#if(count==5):
					#all_file_list.append(file_class)
			'''
			if file_class not in all_file_list and count<5:
				file_path=os.path.join(filename_each,filename)
				print(file_path)
				shutil.copy(file_path,dest_dir)
				os.remove(file_path)
				#all_file_list.append(file_class)
				count=count+1
				if(count==5):
					all_file_list.append(file_class)
			'''
	'''
	all_file_list=[]
	for filename in filelist:
		file_class=filename.split(".")[0].split("_")[0]
		count=0
		if file_class not in all_file_list and count<5:
			file_path=os.path.join(dir,filename)
			shutil.copy(file_path,dest_dir)
			os.remove(file_path)
			#all_file_list.append(file_class)
			count=count+1
			if(count==5):
				all_file_list.append(file_class)
	'''
def load_data_std(dir=h_dir,dest_dir=d_dir):
	if os.path.exists(dest_dir)==False:
		os.mkdir(dest_dir)
	filelist=os.listdir(dir)
	map_file={}
	all_file_list=[]
	for filename in filelist:
		file_class=filename.split(".")[0].split("_")[0]
		map_file[file_class]=0
	for filename in filelist:
		file_class=filename.split(".")[0].split("_")[0]
		if file_class not in all_file_list and map_file[file_class]<5:
				file_path=os.path.join(dir,filename)
				shutil.copy(file_path,dest_dir)
				os.remove(file_path)
				#all_file_list.append(file_class)
				map_file[file_class]=map_file[file_class]+1
				if map_file[file_class]==5:
					all_file_list.append(file_class)
					
def mk_npy(dir):
	rf.getFileArr(dir)
if __name__=="__main__":
	d_dict_path="/media/common/helloworld/train-adversarial/dict-123.txt"
	#rf.getFileArr_std(h_dir,rf.file_label_map(d_dict_path))
	rf.getFileArr_std(d_dir,rf.file_label_map(d_dict_path),False)
