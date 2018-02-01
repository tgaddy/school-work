import monkdata as m
import dtree as d
import drawtree_qt5 as dt
import random
import matplotlib.pyplot as plt
import numpy
import statistics

numValues = {
    0: 3,
    1: 3,
    2: 2,
    3: 3,
    4: 4,
    5: 2
}

monks1_ent = d.entropy(m.monk1)
monks2_ent = d.entropy(m.monk2)
monks3_ent = d.entropy(m.monk3)
'''
print("Entropy for monks 1: ", monks1_ent)
print("Entropy for monks 2: ", monks2_ent)
print("Entropy for monks 3: ", monks3_ent)


print("Monks 1")
for i in range(6):
    print("Avg gain for a%s: ", (i+1))
    print(d.averageGain(m.monk1,m.attributes[i]))

print("Monks 2")
for i in range(6):
    print("Avg gain for a%s: ", (i+1))
    print(d.averageGain(m.monk2,m.attributes[i]))

print("Monks 3")
for i in range(6):
    print("Avg gain for a%s: ", (i+1))
    print(d.averageGain(m.monk3,m.attributes[i]))

# Monks 1
subset1 = d.select(m.monk1,m.attributes[4], 1);
subset2 = d.select(m.monk1,m.attributes[4], 2);
subset3 = d.select(m.monk1,m.attributes[4], 3);
subset4 = d.select(m.monk1,m.attributes[4], 4);


print("Subset 1")
for i in range(6):
    print("Avg gain for attribute ", (i+1))
    print(d.averageGain(subset1,m.attributes[i]))

print("Subset 2")
maxAttr = -1
maxGain = 0.0
for i in range(6):
    curGain = d.averageGain(subset2,m.attributes[i])
    if curGain > maxGain:
        maxGain = curGain
        maxAttr = i
print("Max attribute: ",maxAttr+1, ", gain: ", maxGain)

for i in range(numValues[maxAttr]):
    newSubset = d.select(subset2,m.attributes[maxAttr],i+1)
    print("Value ",i+1, "Most common: ",d.mostCommon(newSubset))

maxAttr = -1
maxGain = 0.0
print("Subset 3")
for i in range(6):
    curGain = d.averageGain(subset3,m.attributes[i])
    if curGain > maxGain:
        maxGain = curGain
        maxAttr = i
print("Max attribute: ",maxAttr+1, ", gain: ", maxGain)

for i in range(numValues[maxAttr]):
    newSubset = d.select(subset3,m.attributes[maxAttr],i+1)
    print("Value ",i+1, "Most common: ",d.mostCommon(newSubset))

maxAttr = -1
maxGain = 0.0
print("Subset 4")
for i in range(6):
    curGain = d.averageGain(subset4,m.attributes[i])
    if curGain > maxGain:
        maxGain = curGain
        maxAttr = i
print("Max attribute: ",maxAttr+1, ", gain: ", maxGain)

for i in range(numValues[maxAttr]):
    newSubset = d.select(subset4,m.attributes[maxAttr],i+1)
    print("Value ",i+1, "Most common: ",d.mostCommon(newSubset))

'''
monk1Tree = d.buildTree(m.monk1,m.attributes)
monk2Tree = d.buildTree(m.monk2,m.attributes)
monk3Tree = d.buildTree(m.monk3,m.attributes)
print("Monk 1 training: ",d.check(monk1Tree,m.monk1))
print("Monk 1 testing: ",d.check(monk1Tree,m.monk1test))
print("Monk 2 training: ",d.check(monk2Tree,m.monk2))
print("Monk 2 testing: ",d.check(monk2Tree,m.monk2test))
print("Monk 3 training: ",d.check(monk3Tree,m.monk3))
print("Monk 3 testing: ",d.check(monk3Tree,m.monk3test))

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def trainTree(data,fraction):
    train, val = partition(data, fraction)

    tree = d.buildTree(train,m.attributes)
    prunedTrees = d.allPruned(tree)
    bestPerfTree = tree
    maxCorrect = 0.0
    for i in range(len(prunedTrees)):
        percentCorrect = d.check(prunedTrees[i],val)
        if percentCorrect > maxCorrect:
            maxCorrect = percentCorrect
            bestPerfTree = prunedTrees[i]

    while d.check(tree,val) <= maxCorrect:
        tree = bestPerfTree
        prunedTrees = d.allPruned(tree)
        maxCorrect = 0.0
        for i in range(len(prunedTrees)):
            percentCorrect = d.check(prunedTrees[i],val)
            if percentCorrect > maxCorrect:
                maxCorrect = percentCorrect
                bestPerfTree = prunedTrees[i]

    return tree

fractions = [0.3,0.4,0.5,0.6,0.7,0.8]
monk1averages = []
monk3averages = []
monk1mins = []
monk1maxs = []
monk3mins = []
monk3maxs = []

for fraction in fractions:
    monk1min = 100
    monk1max = 0.0
    monk1avg = 0.0
    for i in range(100):
        monk1Tree = trainTree(m.monk1,fraction)
        monk1avg += 1-d.check(monk1Tree,m.monk1test)
        if 1-d.check(monk1Tree, m.monk1test) < monk1min:
            monk1min = 1-d.check(monk1Tree, m.monk1test)
        if 1-d.check(monk1Tree, m.monk1test) > monk1max:
            monk1max = 1-d.check(monk1Tree, m.monk1test)

    monk1averages.append(monk1avg/100)
    monk1mins.append(monk1min)
    monk1maxs.append(monk1max)

    monk3min = 100
    monk3max = 0.0
    monk3avg = 0.0
    for i in range(100):
        monk3Tree = trainTree(m.monk3,fraction)
        monk3avg += 1-d.check(monk3Tree,m.monk3test)
        if 1-d.check(monk3Tree, m.monk3test) < monk3min:
            monk3min = 1-d.check(monk3Tree, m.monk3test)
        if 1-d.check(monk3Tree, m.monk3test) > monk3max:
            monk3max = 1-d.check(monk3Tree, m.monk3test)

    monk3averages.append(monk3avg/100)
    monk3mins.append(monk3min)
    monk3maxs.append(monk3max)



plt.plot(fractions,monk1averages,label="Monk1")
plt.plot(fractions,monk3averages,label="Monk3")
plt.xlabel("fractions")
plt.ylabel("Test error")
plt.title("Test error vs fractions")
plt.legend()
plt.show()
