import numpy as np

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	else:
		return 1
		
def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同！
	b = 0.7
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	else:
		return 1
		
def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5]) # 仅权重和偏置与AND不同！
	b = -0.2
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	else:
		return 1

#使用门电路的组合实现XOR
def XOR(x1, x2):
	s1 = NAND(x1, x2)
	s2 = OR(x1, x2)
	y = AND(s1, s2)
	return y
	
print(XOR(1,1),XOR(0,1))