import numpy as np
import random
import matplotlib.pyplot as plt

class PSO(object):
    # 粒子群算法求解最优解(最小值)
    def __init__(self, FitnessFunction, Dimension, Lb, Ub, Vmax, PopulationSize=50, MaxIteration=20, c1=0.4, c2=0.4, w_ini=0.9, w_end=0.4):
        # 优化参数设置
        self.FitnessFunction = FitnessFunction
        self.Dimension = Dimension
        self.Lb = Lb*np.ones((1,self.Dimension),dtype = float)
        self.Ub = Ub*np.ones((1,self.Dimension),dtype = float)
        self.Vmax = Vmax
        self.PopulationSize = PopulationSize
        self.MaxIteration = MaxIteration
        self.c1 = c1
        self.c2 = c2
        self.w_ini = w_ini
        self.w_end = w_end

        # 过程数据保存
        self.x = np.zeros((self.PopulationSize,self.Dimension),dtype=float)
        self.y = np.zeros((self.PopulationSize,1),dtype=float)
        self.v = np.zeros((self.PopulationSize,self.Dimension),dtype=float)
        self.pbest = np.zeros((self.PopulationSize,self.Dimension),dtype=float)
        self.pbest_y = np.zeros((self.PopulationSize,1),dtype = float)
        self.gbest = np.zeros((self.MaxIteration+1,self.Dimension),dtype=float)
        self.gbest_y = np.zeros((self.MaxIteration+1,1),dtype = float)
        self.iter = 0

        # optimize
        self.optimize()
        self.plotOptmizeProcedure()
    
    def optimize(self):
        self.initialPositionVelocity()
        while self.iter <= self.MaxIteration:
            self.fitness()
            self.updatePbest()
            self.updateGbest()
            self.updatePositionVelocity()
            self.saveOptimizeProcedure()
            self.iter += 1
        return

    def plotOptmizeProcedure(self):
        plt.plot(self.gbest_y)
        plt.xlabel('Iteration')
        plt.ylabel('The best fitness')
        plt.show()

    def saveOptimizeProcedure(self):
        if self.iter == 0:
            f = open('./OptimizeProcefure.txt','w')
        else:
            f = open('./OptimizeProcefure.txt','a')
        np.savetxt(f,self.pbest,delimiter=',')
        np.savetxt(f,self.pbest_y,delimiter=',')
        f.write('\n')
        f.close()
        
    def getResult(self):
        return self.gbest_y[self.MaxIteration,:], self.gbest[self.MaxIteration,:]

    # 初始化种群位置、速度
    def initialPositionVelocity(self):
        for i in range(self.PopulationSize):
            self.x[i,:] = self.Lb + (self.Ub-self.Lb)*np.random.random((1,self.Dimension))
            self.v[i,:] = -self.Vmax + 2*self.Vmax*np.random.random((1,self.Dimension))
        return

    def fitness(self):
        for i in range(self.PopulationSize):
            self.y[i,:] = self.FitnessFunction(self.x[i,:],self.Dimension)
        return

    def updatePbest(self):
        if self.iter == 0:
            self.pbest = self.x
            self.pbest_y = self.y
        else:
            for i in range(self.PopulationSize):
                if self.y[i,:] < self.pbest_y[i,:]:
                    self.pbest[i,:] = self.x[i,:]     
                    self.pbest_y[i,:] = self.y[i,:] 
        return

    def updateGbest(self):
        pbestmin = np.min(self.pbest_y,axis = 0)
        ID = np.where(self.pbest_y==pbestmin)
        if self.iter == 0:
            self.gbest[0,:] = self.pbest[ID[0],:]
            self.gbest_y[0,:] = self.pbest_y[ID[0][0],:]
        else:
            if pbestmin < self.gbest_y[self.iter-1,:]:
                self.gbest[self.iter,:] = self.pbest[ID[0][0],:]
                self.gbest_y[self.iter,:] = pbestmin
            else:
                self.gbest[self.iter,:] = self.gbest[self.iter-1,:]
                self.gbest_y[self.iter,:] = self.gbest_y[self.iter-1,:]
   
    def updatePositionVelocity(self):
        # 更新位置速度
        for i in range(self.PopulationSize):
            w_k = (self.w_ini-self.w_end)*(self.MaxIteration-self.iter)/self.MaxIteration+self.w_end
            self.v[i,:] = w_k*self.v[i,:] + self.c1*random.random()*(self.pbest[i,:]-self.x[i,:])+self.c2*random.random()*(self.gbest[self.iter,:]-self.x[i,:])
            self.x[i,:] = self.x[i,:] + self.v[i,:]
        # 限制位置速度边界
        for j in range(self.PopulationSize):
            for k in range(self.Dimension):
                # 限制位置
                if self.x[j,k] < self.Lb[0,k]:
                    self.x[j,k] = self.Lb[0,k]
                elif self.x[j,k] > self.Ub[0,k]:
                    self.x[j,k] = self.Ub[0,k]
                # 限制速度
                if self.v[j,k] < -self.Vmax:
                    self.v[j,k] = -self.Vmax
                elif self.v[j,k] > self.Vmax:
                    self.v[j,k] = self.Vmax
        return

