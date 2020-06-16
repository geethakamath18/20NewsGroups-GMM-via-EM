#!/usr/bin/env python
# coding: utf-8

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sklearn.metrics
from scipy.stats import multivariate_normal


# In[39]:


def linear_transform(a, e):
    assert a.ndim == 1
    assert np.allclose(1, np.sum(e**2))
    u = a - np.sign(a[0]) * np.linalg.norm(a) * e  
    v = u / np.linalg.norm(u)
    H = np.eye(len(a)) - 2 * np.outer(v, v)
    return H


# In[40]:


def QR(matrix):    
    n,m=matrix.shape #TALL-WIDE
    assert n >= m  
    Q = np.eye(n)
    R = matrix.copy()
    for i in range(m - int(n==m)):
        r = R[i:, i]
        if np.allclose(r[1:], 0):
            continue   
        # e is the i-th basis vector of the minor matrix.
        e = np.zeros(n-i)
        e[0] = 1  
        H = np.eye(n)
        H[i:, i:] = linear_transform(r, e)
        Q = Q @ H.T
        R = H @ R   
    return Q, R


# In[41]:


class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = bool(whiten)
    
    def fit(self, X):
        n, m = X.shape
        self.mu = X.mean(axis=0)
        X = X - self.mu
        C = X.T @ X / (n-1) #Eigen Decomposition
        C_k = C
        Q_k = np.eye( C.shape[1] )
        for k in range(100): #number of iterations=100
            Q, R = QR(C_k)
            Q_k = Q_k @ Q
            C_k = R @ Q
        self.eigenvalues =  np.diag(C_k)
        self.eigenvectors = Q_k
        if self.n_components is not None:  # truncate the number of components
            self.eigenvalues = self.eigenvalues[0:self.n_components]
            self.eigenvectors = self.eigenvectors[:, 0:self.n_components]      
        descending_order = np.flip(np.argsort(self.eigenvalues)) #eigenvalues in descending order
        self.eigenvalues = self.eigenvalues[descending_order]
        self.eigenvectors = self.eigenvectors[:, descending_order]
        return self

    def transform(self, X):
        X = X - self.mu
        if self.whiten:
            X = X / self.std
        return X @ self.eigenvectors


# In[42]:


newsgroups_train1 = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes')) #without metadata


# In[43]:


newsgroups_test = fetch_20newsgroups(subset='test')


# In[44]:


categories = newsgroups_train1.target_names


# In[45]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,max_features=200, stop_words='english')


# In[46]:


afterTFIDF1 = tfidf_vectorizer.fit_transform(newsgroups_train1.data)


# In[47]:


training1=afterTFIDF1.toarray()


# In[74]:


training1[0]


# In[49]:


pca1 = PCA(whiten=False, n_components=2)

pca1.fit(training1)
final1 = pca1.transform(training1)


# In[50]:


final1.shape #theta


# In[51]:


gmm1 = GaussianMixture(n_components=20,covariance_type='full',max_iter=100,verbose=1)


# In[52]:


gmm1.fit(training1)


# In[53]:


means1=gmm1.means_


# In[67]:


y=training1 @ means1.T #tehta into muc


# In[68]:


y.shape


# In[65]:


y.sort(axis=0)


# In[66]:


y


# In[79]:


ind1 = np.argpartition(y[:,0], -10)[-10:]
for i in ind1:
    print(newsgroups_train1.target[i])


# In[80]:


ind2 = np.argpartition(y[:,1], -10)[-10:]
for i in ind2:
    print(newsgroups_train1.target[i])


# In[81]:


ind3 = np.argpartition(y[:,2], -10)[-10:]
for i in ind3:
    print(newsgroups_train1.target[i])


# In[82]:


ind4 = np.argpartition(y[:,3], -10)[-10:]
for i in ind4:
    print(newsgroups_train1.target[i])


# In[83]:


ind5 = np.argpartition(y[:,4], -10)[-10:]
for i in ind5:
    print(newsgroups_train1.target[i])


# In[84]:


ind6 = np.argpartition(y[:,5], -10)[-10:]
for i in ind6:
    print(newsgroups_train1.target[i])


# In[85]:


ind7 = np.argpartition(y[:,6], -10)[-10:]
for i in ind7:
    print(newsgroups_train1.target[i])


# In[86]:


ind8 = np.argpartition(y[:,7], -10)[-10:]
for i in ind8:
    print(newsgroups_train1.target[i])


# In[87]:


ind9 = np.argpartition(y[:,8], -10)[-10:]
for i in ind9:
    print(newsgroups_train1.target[i])


# In[88]:


ind10 = np.argpartition(y[:,9], -10)[-10:]
for i in ind10:
    print(newsgroups_train1.target[i])


# In[89]:


ind11 = np.argpartition(y[:,10], -10)[-10:]
for i in ind11:
    print(newsgroups_train1.target[i])


# In[90]:


ind12 = np.argpartition(y[:,11], -10)[-10:]
for i in ind12:
    print(newsgroups_train1.target[i])


# In[91]:


ind13 = np.argpartition(y[:,12], -10)[-10:]
for i in ind13:
    print(newsgroups_train1.target[i])


# In[92]:


ind14 = np.argpartition(y[:,13], -10)[-10:]
for i in ind14:
    print(newsgroups_train1.target[i])


# In[93]:


ind15 = np.argpartition(y[:,14], -10)[-10:]
for i in ind15:
    print(newsgroups_train1.target[i])


# In[94]:


ind16 = np.argpartition(y[:,15], -10)[-10:]
for i in ind16:
    print(newsgroups_train1.target[i])


# In[95]:


ind17 = np.argpartition(y[:,16], -10)[-10:]
for i in ind17:
    print(newsgroups_train1.target[i])


# In[96]:


ind18 = np.argpartition(y[:,17], -10)[-10:]
for i in ind18:
    print(newsgroups_train1.target[i])


# In[97]:


ind19 = np.argpartition(y[:,18], -10)[-10:]
for i in ind19:
    print(newsgroups_train1.target[i])


# In[98]:


ind20 = np.argpartition(y[:,19], -10)[-10:]
for i in ind20:
    print(newsgroups_train1.target[i])


# # Even though most of the prediction in any given group correspond to a single cluster, there are still prediction errors, therefore, it does not make sense

# In[ ]:




