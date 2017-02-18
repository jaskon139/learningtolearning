# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

coordinates = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
                        [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
                        [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
                        [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
                        [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
                        [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],
                        [420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],
                        [685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],
                        [475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],
                        [830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],
                        [1340.0,725.0],[1740.0,245.0]])





def getdistmat(coordinates):
    num = coordinates.shape[0]
    print "num= %d \n" %( num )
    distmat = np.zeros((num,num))
    for i in range(num):
        for j in range(i,num):
            distmat[i][j] = distmat[j][i]=np.linalg.norm(coordinates[i]-coordinates[j])
    return distmat

class yuqun:
    def __init__(self, coord = coordinates, numantall = 40):
        self.numant = numantall #蚂蚁个数
        self.numcity = coord.shape[0] #城市个数
        self.pheromonetable  = np.ones((self.numcity,self.numcity)) # 信息素矩阵
        self.Q = 1
        self.rho = 0.1  # 信息素的挥发速度
        self.alpha = 1   #信息素重要程度因子
        self.distmat = getdistmat(coord)  # 城市的距离矩阵
        self.etatable = 1.0/(self.distmat+ np.diag([1e10]*self.numcity)) #启发函数矩阵，表示蚂蚁从城市i转移到矩阵j的期望程度
        self.iter =0
        return

    def next_step(self, lengthaver, lengthbest, pathbest):

        pathtable = np.zeros((self.numant,self.numcity)).astype(int) #路径记录表
        # 随机产生各个蚂蚁的起点城市
        if self.numant <= self.numcity:#城市数比蚂蚁数多
            pathtable[:,0] = np.random.permutation(range(0,self.numcity))[:self.numant]
        else: #蚂蚁数比城市数多，需要补足
            pathtable[:self.numcity,0] = np.random.permutation(range(0,self.numcity))[:]
            pathtable[self.numcity:,0] = np.random.permutation(range(0,self.numcity))[:self.numant-self.numcity]

        length = np.zeros(self.numant) #计算各个蚂蚁的路径距离

        for i in range(self.numant):
            visiting = pathtable[i,0] # 当前所在的城市

            #visited = set() #已访问过的城市，防止重复
            #visited.add(visiting) #增加元素
            unvisited = set(range(self.numcity))#未访问的城市
            unvisited.remove(visiting) #删除元素

            for j in range(1,self.numcity):#循环numcity-1次，访问剩余的numcity-1个城市

                #每次用轮盘法选择下一个要访问的城市
                listunvisited = list(unvisited)

                probtrans = np.zeros(len(listunvisited))

                for k in range(len(listunvisited)):
                    probtrans[k] = np.power(self.pheromonetable[visiting][listunvisited[k]],self.alpha) \
                                   *np.power(self.etatable[visiting][listunvisited[k]],self.alpha)
                cumsumprobtrans = (probtrans/sum(probtrans)).cumsum()

                cumsumprobtrans -= np.random.rand()

                k = listunvisited[np.where(cumsumprobtrans>0)[0][0]] #下一个要访问的城市

                pathtable[i,j] = k

                unvisited.remove(k)
                #visited.add(k)

                length[i] += self.distmat[visiting][k]

                visiting = k

            length[i] += self.distmat[visiting][pathtable[i,0]] #蚂蚁的路径距离包括最后一个城市和第一个城市的距离


        #print length
        # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数

        lengthaver[self.iter] = length.mean()

        if self.iter == 0:
            lengthbest[self.iter] = length.min()
            pathbest[self.iter] = pathtable[length.argmin()].copy()
        else:
            if length.min() > lengthbest[self.iter-1]:
                lengthbest[self.iter] = lengthbest[self.iter-1]
                pathbest[self.iter] = pathbest[self.iter-1].copy()

            else:
                lengthbest[self.iter] = length.min()
                pathbest[self.iter] = pathtable[length.argmin()].copy()


        # 更新信息素
        changepheromonetable = np.zeros((self.numcity,self.numcity))
        for i in range(self.numant):
            for j in range(self.numcity-1):
                changepheromonetable[pathtable[i,j]][pathtable[i,j+1]] += self.Q/self.distmat[pathtable[i,j]][pathtable[i,j+1]]

            changepheromonetable[pathtable[i,j+1]][pathtable[i,0]] += self.Q/self.distmat[pathtable[i,j+1]][pathtable[i,0]]

        self.pheromonetable = (1-self.rho)*self.pheromonetable + changepheromonetable


        self.iter += 1 #迭代次数指示器+1

        #观察程序执行进度，该功能是非必须的
        if (self.iter-1)%20==0:
            print self.iter-1

        return self.iter,  lengthaver, lengthbest, pathbest;

def main(_):

    #self.numcity = 2000
    #self.numcity = coordinates.shape[0]

    #coordinates = np.zeros((self.numcity,2 ))

    #print coordinates

    #for i in range(self.numcity):
    #    coordinates[i][0]= random.randint(10, 4000)
    #    coordinates[i][1] =random.randint(10, 3900)

    #print coordinates
    itermax = 250
    numcity = coordinates.shape[0]

    lengthaver = np.zeros(itermax) #各代路径的平均长度
    lengthbest = np.zeros(itermax) #各代及其之前遇到的最佳路径长度
    pathbest = np.zeros((itermax,numcity)) # 各代及其之前遇到的最佳路径长度
    iter = 0
    calculate = yuqun()

    while iter < itermax:
        iter, lengthaver, lengthbest, pathbest  \
            = calculate.next_step( lengthaver, lengthbest, pathbest)

        if (iter - 1) % 10 == 0:
            # 做出平均路径长度和最优路径长度
            fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,10))
            axes[0].plot(lengthaver,'k',marker = u'')
            axes[0].set_title('Average_Best%d.png'%(iter-1))
            axes[0].set_xlabel(u'iteration')

            axes[1].plot(lengthbest,'k',marker = u'')
            axes[1].set_title('BestPath%d.png'%(iter-1))
            axes[1].set_xlabel(u'iteration')
            fig.savefig('Average_Best%d.png'%(iter-1),dpi=500,bbox_inches='tight')
            plt.close()

            #作出找到的最优路径图
            print  pathbest
            bestpath = pathbest[iter-1]
            plt.plot(coordinates[:,0],coordinates[:,1],'r.',marker=u'$\cdot$')
            plt.xlim([-100,4000])
            plt.ylim([-100,4000])

            for i in range(numcity-1):#
                m,n = bestpath[i],bestpath[i+1]
                print m,n
                plt.plot([coordinates[m][0],coordinates[n][0]],[coordinates[m][1],coordinates[n][1]],'k')
            plt.plot([coordinates[bestpath[0]][0],coordinates[n][0]],[coordinates[bestpath[0]][1],coordinates[n][1]],'b')

            ax=plt.gca()
            ax.set_title('BestPath%d.png'%(iter-1))
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y_axis')

            plt.savefig('BestPath%d.png'%(iter-1),dpi=500,bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    tf.app.run()
