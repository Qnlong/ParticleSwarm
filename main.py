import math
import ParticleSwarm.ParticleSwarm as PSO

def Griewank(x,N):
    sum = 0
    product = 1
    for i in range(N):
        sum += x[i]**2/4000
        product *= math.cos(x[i]/math.sqrt(i+1))
    return sum - product + 1

# test
if __name__ == '__main__':
    PSO1 = PSO.PSO(Griewank,20,-600,600,1200,30,500,2,2)
    print(PSO1.getResult())
