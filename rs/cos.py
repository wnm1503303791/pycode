list1 = [ 'runoob', 786 , 2.23, 'john', 70.2 ]

print(type(list1[0]),type(list1[1]),type(list1[2]))

list2=[
[1 ,0.849703 ,0.676155 ,-0.337082 ,0.908417],
[2 ,0.800687 ,0.633915 ,-0.780155 ,1.38767]
]

print(type(list2[0]),list2[0])

import numpy as np
def readfile(path):
	f=open(path)
	flag=True #用于处理第一行数据
	for data in f.readlines():
		data=data.strip("\n")
		fla=data.split(" ")
		fla=fla[1:]
		if flag:
			fla=[float(x) for x in fla]	#语法糖，将新读入的数据行内的所有元素从字符串转换为float
			matrix=np.array(fla)
			flag=False
		else:
			fla=[float(x) for x in fla]
			matrix=np.c_[matrix,fla]
	dealMatrix(matrix)
	f.close()
	return matrix
	
def dealMatrix(matrix):
    print("transpose the matrix")
    matrix = matrix.transpose()
    print(matrix)

    print("matrix trace ")
    print(np.trace(matrix))


data=readfile("aa_at_pre4_sort.txt")
data=data.transpose()
print(data[0],data[0]*data[0])
#op7=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
print(np.dot(data[0],data[0])/(np.linalg.norm(data[0])*(np.linalg.norm(data[0]))))