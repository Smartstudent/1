import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#读取文件数据
def loaddata(filename):
	dataset = []
	fp = open(filename)
	for i in fp.readlines():

		#数据集比较庞大，在此更更改使用的数据数量以此对比
		if len(dataset) >= 1000:
			break
		a = i.strip().split()

		#只是用其中位置为3，4，5的数据，形成一个三维数据，便于可视化
		dataset.append([float(a[3]), float(a[4]), float(a[5])])
	return dataset

#可视化数据，绘制回归曲线
def Data_plot(data, w):
	x = data[:,0]
	y = data[:,1]
	z = data[:,2]

	x1 = [0, 600]
	y1 = [0, 600]
	z1 = [w[2, 0], 600*w[0, 0]+600*w[1, 0]+w[2, 0]]
	 
	# 绘制散点图
	figer = plt.figure()
	ax = Axes3D(figer)
	ax.scatter(x, y, z)
	ax.plot(x1, y1, z1, 'r', color = 'black')
	 
	# 添加坐标轴(顺序是Z, Y, X)
	ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
	ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
	ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
	plt.show()

if __name__ == '__main__':
	#读取文件
	dataset = loaddata('数据集.txt')

	#提取属性，数据中前两列作为属性，最后一列作为标准特征度量
	x = np.mat(dataset)[:,(0,1)]
	y = np.mat(dataset)[:,2]

	#调整属性矩阵（在原矩阵最后一列添加值为1的列）
	a = np.ones((len(dataset), 1))
	x = np.c_[x, a]

	#计算属性矩阵x的转置
	xt = np.transpose(x)

	#按照计算式计算参数w，np.dot函数用于计算矩阵乘法
	a = np.dot(xt, x).I
	b = np.dot(a, xt)
	w = np.dot(b, y)

	#绘制回归函数曲线和散点图，不用掌握
	Data_plot(np.mat(dataset), w)
