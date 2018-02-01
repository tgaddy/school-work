from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math

numpy.random.seed(100)
random.seed(100)

classA = [(random.normalvariate(-1.5,1),
           random.normalvariate(0.5,1),
           1.0)
           for i in range(5)] + \
         [(random.normalvariate(1.5,1),
           random.normalvariate(0.5,1),
           1.0)
           for i in range(5)]

classB = [(random.normalvariate(0.0,0.5),
           random.normalvariate(-0.5,0.5),
           -1.0)
           for i in range(10)]

data = classA + classB
random.shuffle(data)



def linearKernel(xi, xj):
    return numpy.dot(xi[:2],xj[:2]) + 1

def polynomialKernel(xi,xj):
    p = 2
    return linearKernel(xi,xj)**p

def kernelFn(xi,xj,kernelName):
    if kernelName == "lin":
        return linearKernel(xi,xj)
    elif kernelName == "pol":
        return polynomialKernel(xi,xj)


def initializeP(dataVals):
    dimSize = len(dataVals)
    P = numpy.empty((dimSize,dimSize))
    for i in range(dimSize):
        for j in range(dimSize):
            P[i][j] = dataVals[i][2]*dataVals[j][2]*kernelFn(dataVals[i],dataVals[j],"lin")
    return P

def indicator(xStar,alpha):
    sum = 0
    for i in range(len(alpha)):
        xi = alpha[i][1]
        sum += alpha[i][0]*xi[2]*kernelFn(xStar,xi,"lin")
    return sum


P = initializeP(data)
G = numpy.identity(len(data)) * -1
q = numpy.empty(len(data))
q.fill(-1)
#q.transpose()
h = numpy.zeros(len(data))

r = qp (matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])
alpha = numpy.array(alpha)
nonZeroAlpha = alpha[alpha>0.00001]

print(alpha)
print(len(alpha))

print(nonZeroAlpha)
print(len(nonZeroAlpha))

result = [(alpha[i],data[i]) for i in range(len(data)) if alpha[i]>0.00001]
print("result:",result)

#pylab.hold(True)
pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo')
pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro')

#pylab.show()
