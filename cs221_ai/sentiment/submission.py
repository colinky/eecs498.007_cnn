#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return dict(collections.Counter(x.split()))
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(x):
        return -1 if dotProduct(weights, featureExtractor(x)) < 0 else +1 
    for _ in range(numIters):
        train_size = len(trainExamples)
        rnd_spl = random.sample(range(train_size), k=train_size)             
        for pos in rnd_spl:
            x, y = trainExamples[pos]
            phix = featureExtractor(x)
            if dotProduct(weights, phix) * y < 1:
                increment(weights, eta*y, phix)
        # print(f'for training set: {evaluatePredictor(trainExamples, predictor):.4f}, for test set: {evaluatePredictor(testExamples, predictor):.4f}')
   # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
        phi[list(weights)[random.randint(0, len(weights)-1)]] = random.random()
        # phi[''.join(random.sample(list(weights), k=1))] = random.random()
        y = -1 if dotProduct(weights, phi) < 0 else +1       
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.replace(' ', '')
        dic = dict()
        for i in range(len(x)-n+1):
            if x[i:i+n] in dic: dic[x[i:i+n]] += 1
            else: dic[x[i:i+n]] = 1
        return dic
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    centroids = list(random.sample(examples, k=K))
    assignments = []
    prev_loss = 1e10
    
    empl2 = []
    for empl in examples: 
        empl2.append(dotProduct(empl, empl))

    for _ in range(maxIters):
        ctrd2 = []  
        for ctrd in centroids: 
            ctrd2.append(dotProduct(ctrd, ctrd))

        assignments, tmp_loss = list(), 0
        for cnt_empl, empl in enumerate(examples):
            idx_min_dist, min_distance2 = -1, 1e10
            for cnt_ctr, ctr in enumerate(centroids):
                distance2 = empl2[cnt_empl] + ctrd2[cnt_ctr] - 2 * dotProduct(ctr, empl)
                if min_distance2 > distance2:
                    idx_min_dist = cnt_ctr
                    min_distance2 = distance2                 
            assignments.append(idx_min_dist)
            tmp_loss += min_distance2
        
        ctrds_new = [None]*K
        cnter_asgn = Counter(assignments)
        for idx, asgn in enumerate(assignments):
            if ctrds_new[asgn] == None: 
                ctrds_new[asgn] = examples[idx].copy()
            else:
                increment(ctrds_new[asgn], 1, examples[idx])

        for idx, ctrd in enumerate(ctrds_new):
            for k, v in ctrd.items():
                ctrd[k] /= cnter_asgn[idx]

        if prev_loss == tmp_loss:
            return centroids, assignments, tmp_loss
        else:
            prev_loss = tmp_loss
            centroids = ctrds_new

    return centroids, assignments, prev_loss    
    # END_YOUR_CODE
