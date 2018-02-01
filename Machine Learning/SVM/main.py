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
           random.normalvariate(-1.5,0.5),
           -1.0)
           for i in range(10)]

data = classA + classB
random.shuffle(data)
usedKernel = "sig"
C = 5

def linearKernel(xi, xj):
    return (numpy.dot(xi[:2],xj[:2]) + 1)

def polynomialKernel(xi,xj):
    p = 2
    return (numpy.dot(xi[:2],xj[:2]) + 1)**p
    #return linearKernel(xi,xj)**p

def radialKernel(xi,xj):
    sigma = 5
    #print("val: ", math.exp(-pow(numpy.linalg.norm(xi[:2]-xj[:2]),2)/(2*pow(sigma,2))))
    return math.exp(-pow(numpy.linalg.norm(xi[:2]-xj[:2]),2)/(2*pow(sigma,2)))

def sigmoidKernel(xi,xj):
    k = .1
    delta = 0
    #xi[:2] *= k
    # print("val:",math.tanh(numpy.dot(xi[:2],xj[:2]) - delta))
    return math.tanh(numpy.dot(k*xi[:2],xj[:2]) - delta)

def kernelFn(xi,xj,kernelName):
    if kernelName == "lin":
        return linearKernel(xi,xj)
    elif kernelName == "pol":
        return polynomialKernel(xi,xj)
    elif kernelName == "rad":
        return radialKernel(xi,xj)
    elif kernelName == "sig":
        return sigmoidKernel(xi,xj)
    else:
        print("Type in a valid kernel name")


def initializeP(data):
    dimSize = len(data)
    P = numpy.empty((dimSize,dimSize))
    for i in range(dimSize):
        for j in range(dimSize):
            P[i][j] = data[i][2]*data[j][2]*kernelFn(numpy.array(data[i]),numpy.array(data[j]),usedKernel)
    return P

def indicator(xStar,alpha):
    sum = 0
    for i in range(len(alpha)):
        xi = numpy.array(alpha[i][1])
        sum += alpha[i][0]*xi[2]*kernelFn(xStar,xi,usedKernel)
    return sum

P = initializeP(data)
G = numpy.identity(len(data)) * -1
G = numpy.concatenate((G,numpy.identity(len(data))),axis=0)
print(G)
q = numpy.empty(len(data))
q.fill(-1)
q.transpose()
#h = numpy.zeros(len(data))
h = numpy.concatenate((numpy.zeros(len(data)), numpy.full((len(data),),C)),axis=0)
print(h)

r = qp(matrix(P), matrix(q), matrix(G), matrix(h)) #, kktsolver='ldl')
alpha = list(r['x'])
alpha = numpy.array(alpha)


result = [(alpha[i],data[i]) for i in range(len(data)) if alpha[i]>0.00001]
print("result:",result)


pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo')
pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro')

pylab.plot([p[1][0] for p in result if p[1] in classA],
           [p[1][1] for p in result if p[1] in classA],
           'bx')
pylab.plot([p[1][0] for p in result if p[1] in classB],
           [p[1][1] for p in result if p[1] in classB],
           'rx')

xrange=numpy.arange(-4, 4, 0.05)
yrange=numpy.arange(-4, 4, 0.05)
grid=matrix([[indicator(numpy.array([x, y]),result)
            for y in yrange ]
            for x in xrange ])
pylab.contour(xrange, yrange, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1, 3, 1))
pylab.show()
