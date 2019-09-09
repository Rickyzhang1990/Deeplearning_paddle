import os,sys
from PIL import Image, ImageFilter
import random




def Getimg(train_list):
	with open(train_list) as f:
		for line in f:
			img = line.strip().split("\t")[0]
			absimg = os.path.abspath(img)
			cls = line.strip().split("\t")[1]
			yield absimg,cls
			
			
def modify_file_path(file_path, middle_name):
	# file_path = "D:/test/test.jpy"
	# middle_name = 'Hor'
	(filepath,tempfilename) = os.path.split(file_path)
	# print(filepath,tempfilename)
	(filename,extension) = os.path.splitext(tempfilename)
	# print(filename,extension)
	filename = filename + '_' + middle_name + extension
	file_path = os.path.join(filepath, filename)
	# file_path = 'D:/test/test_Hor.jpy'
	return file_path
			
#水平翻转
def Hor(file_abs_path):	
	middle_name = 'Hor'
	new_file_path = modify_file_path(file_abs_path, middle_name)
	im = Image.open(file_abs_path).convert('RGB')
	im.transpose(Image.FLIP_LEFT_RIGHT).save(new_file_path)
	
	return new_file_path
	
def RandCrop(file_abs_path):
	randint = random.randint(1, 10)
	middle_name = 'RandCrop_%d' % (randint) 
	new_file_path = modify_file_path(file_abs_path, middle_name)
	
	im = Image.open(file_abs_path).convert('RGB')
	width, height = im.size
	#print(width, height)
	ratio = 0.88    #0.8--0.9之间的一个数字
	left = int(width*(1-ratio)*random.random())   #左上角点的横坐标
	top = int(height*(1-ratio)*random.random())   #左上角点的纵坐标

	crop_img = (left, top, left+width*ratio, top+height*ratio)
	im_RCrops = im.crop(crop_img)
	im_RCrops.save(new_file_path)
	return new_file_path
	
#色彩抖动
def Jittering(file_abs_path):
	im = Image.open(file_abs_path).convert('RGB')
	i_randint = random.randint(1, 9)
	
	if i_randint==1:
	# 高斯模糊
		middle_name = 'Jittering_%s' % ('GaussianBlur') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.GaussianBlur).save(new_file_path)
	elif i_randint==2:
	# 普通模糊		
		middle_name = 'Jittering_%s' % ('BLUR') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.BLUR).save(new_file_path)		
	elif i_randint==3:
	# 边缘增强
		middle_name = 'Jittering_%s' % ('EDGE_ENHANCE') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.EDGE_ENHANCE).save(new_file_path)			
	elif i_randint==4:
	# 找到边缘
		middle_name = 'Jittering_%s' % ('FIND_EDGES') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.FIND_EDGES).save(new_file_path)	
	elif i_randint==5:
	# 浮雕
		middle_name = 'Jittering_%s' % ('EMBOSS') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.EMBOSS).save(new_file_path)
	elif i_randint==6:
	# 轮廓
		middle_name = 'Jittering_%s' % ('CONTOUR') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.CONTOUR).save(new_file_path)
	elif i_randint==7:
	# 锐化
		middle_name = 'Jittering_%s' % ('SHARPEN') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.SHARPEN).save(new_file_path)
	elif i_randint==8:
	# 平滑
		middle_name = 'Jittering_%s' % ('SMOOTH') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.SMOOTH).save(new_file_path)
	else:
	# 细节
		middle_name = 'Jittering_%s' % ('DETAIL') 
		new_file_path = modify_file_path(file_abs_path, middle_name)		
		im.filter(ImageFilter.DETAIL).save(new_file_path)
		
	return new_file_path
	
def augfie(train_list):
	with open(train_list + "_2.txt" ,'w') as newtrain:
		for img ,cls in Getimg(train_list):
#			print(img)
			nimg1 = Hor(img)
#			print(nimg1)
			nimg2 = RandCrop(img)
			nimg3 = RandCrop(img)
			nimg4 = Jittering(img)
			nimg5 = Jittering(img)
#			nimg6 = RandCrop(Hor(img))
			lst = [nimg1 + "\t" + cls ,nimg2 + "\t" + cls , nimg3 + "\t" + cls , nimg4 + "\t" + cls , nimg5 + "\t" + cls ]
			for x in lst:
				newtrain.write(x + "\n")
				
				
				
				
if __name__ == "__main__":
	x = sys.argv[1].strip()
	augfie(x)