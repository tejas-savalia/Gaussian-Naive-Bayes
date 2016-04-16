
# coding: utf-8

# In[1]:

import numpy


# Preprocessing of the breast cancer database. The rows with missing values are removed

# In[2]:

def preprocessing(url):
    loaded = numpy.genfromtxt(url, dtype = str, missing_values = '?')
    X1 = list()
    X = list()
    y = list()
    for i in loaded:
        if '?' not in i:
            X1.append(i.split(','))
    for i in X1:
        X.append(i[1:-1])
        y.append(i[-1])
    X = numpy.array(X).astype(int)
    y = numpy.array(y).astype(int)
    y = numerize_features(y)
    return (X, y)


# Convert string categorical features to number of categories

# In[3]:

def numerize_features(a):
    distinct = numpy.unique(a)
    column = numpy.zeros(len(a))
    distinct = distinct.tolist()
    #print loaded[:, 1] == distinct[1]
    for i in range(len(a)):
        column[i] = distinct.index(a[i])
    return column.astype(int)


# In[4]:

(X, y) = preprocessing("E:\Lecs\IIIT\SMAI\Assignments\Assignment 4\Breast cancer dataset\\breast-cancer-wisconsin.data.txt")


# In[5]:

def group_data(X, y):
    X_temp1 = numpy.empty((len(X[1, :])))
    X_temp0 = numpy.empty((len(X[1, :])))
    print X_temp0
    for i, j in zip(X, y):
        if j == 1:
            X_temp1 = numpy.column_stack((X_temp1, i))
        else:
            X_temp0 = numpy.column_stack((X_temp0, i))
    return (X_temp0, X_temp1)
(grouped0, grouped1) = group_data(X, y)
grouped0, grouped1 = grouped0[:, 1:], grouped1[:, 1:]


# In[6]:

print numpy.shape(grouped1)
print numpy.shape(grouped0)
print numpy.shape(X)


# In[7]:

def variance(grouped0, grouped1):
    a = numpy.var(grouped0, axis = 1)
    b = numpy.var(grouped1, axis = 1)
    variance = numpy.array((a, b))
    return variance
variance = variance(grouped0, grouped1)
print variance


# Calculate prior probabilities for all classes

# In[8]:

prior = numpy.log(numpy.bincount(y)/float(len(y)))
print prior


# Mean of attribute in a given class

# In[9]:

def mean_attr(f, w):
    #mean = numpy.mean(f)
    mean = numpy.zeros((len(f[0, :]), len(numpy.unique(w))))
    for i, j in zip(f, w):
        mean[:, j] = mean[:, j] + i

    mean[:, 0] = mean[:, 0]/float(numpy.bincount(w)[0])
    mean[:, 1] = mean[:, 1]/float(numpy.bincount(w)[1])
    return mean
    

mean = mean_attr(X, y)


# In[10]:

print mean


# In[11]:

print numpy.shape(variance)
print numpy.shape(mean)
variance = variance.T


# Calculate the likelihood ratio for each feature given a class

# In[12]:

def calculate_likelihood_gaussian(mean, variance, v, f, w):
    gauss = (1/numpy.sqrt(2*3.14*variance[f][w]))*numpy.exp(-numpy.square((v - mean[f][w]))/(2*variance[f][w]))
    return numpy.log(gauss)


# In[13]:

print calculate_likelihood_gaussian(mean, variance, 0.5, 4, 1)


# In[ ]:




# In[35]:

def calculate_posterior(prior, v, w):
    posterior = numpy.log(1)
    #print f
    for i in range(len(v)):
        #print likelihood[i][f[i]][w]
        likelihood = calculate_likelihood_gaussian(mean, variance, v[i], i, w)
        posterior = posterior + likelihood
    return (prior[w] + posterior)


# In[40]:

likelihood = calculate_posterior(prior, X[98, :], 1)
print likelihood


# In[44]:

def Naive_Bayes_predict(prior, f, w):
    posteriors = numpy.zeros((len(w)))
    for i in range(len(w)):
        posteriors[i] = calculate_posterior(prior, f, i)
    return numpy.argmax(posteriors)


# In[45]:

print Naive_Bayes_predict(prior,  X[98, :], numpy.unique(y))
y[98]


# In[49]:

def accuracy(XTest, yTest):
    count = 0
    for i in range(len(yTest)):
        if Naive_Bayes_predict(prior, XTest[i, :], numpy.unique(y)) == yTest[i]:
            count = count + 1
    return count/float(len(yTest))


# In[51]:

print accuracy(X, y)


# In[ ]:



