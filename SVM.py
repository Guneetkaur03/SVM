
# coding: utf-8

# In[111]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from matplotlib import style
from sklearn.metrics import accuracy_score


# In[112]:


df = pd.read_csv('C:/Users/Student/Desktop/7Th Sem/ML/data.csv')
df.head()
plt.rcParams['figure.figsize'] = (12, 6)
style.use('ggplot')


# In[113]:


#X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
#y = df['target'].values[np.newaxis,:]
X = df[['x1','x2']]
y = df['target']
print X.shape
print y.shape


# In[114]:


def draw_svm(X, y, C):
    # Plotting the Points
    plt.scatter(df['x1'], df['x2'], c=y)
    
    # The SVM Model with given C parameter
    clf = SVC(kernel='linear', C=C)
    clf_fit = clf.fit(X,y)
    
    # Limit of the axes
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Creating the meshgrid
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # Plotting the boundary
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
                        alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], 
                clf.support_vectors_[:, 1], 
                s=100, linewidth=1, facecolors='none')
    plt.show()
    # Returns the classifier
    return clf_fit


# In[117]:


clf_arr = []
clf_arr.append(draw_svm(X, y, 0.01))
clf_arr.append(draw_svm(X, y, 0.1))
clf_arr.append(draw_svm(X, y, 1))
clf_arr.append(draw_svm(X, y, 10))

for i, clf in enumerate(clf_arr):
    score1 = clf.score(X,y)
    print score1

