
# coding: utf-8

# ## Randomforest

# In[10]:


import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model        
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten

num_pairs = 3000

pos_dir = '../16_8/positive_x/*'
neg_dir = '../16_8/negative_x/*'

pro_name =list()
lig_name = list()

pos_files_path = sorted(glob.glob(pos_dir))
neg_files_path = sorted(glob.glob(neg_dir))

# Load first positive sample
X = np.loadtxt(pos_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X = X[np.newaxis,:,:,np.newaxis]
# Concat first negative sample
X_neg = np.loadtxt(neg_files_path[0], delimiter=',', dtype=np.float64, ndmin=2)
X_neg = X_neg[np.newaxis,:,:,np.newaxis]
X = np.concatenate((X,X_neg), axis=0)

for i in range(0,num_pairs-1):
    text = str(pos_files_path[i])
    number =text[19:28]
    pro_name.append(number.split("_")[0])
    lig_name.append(number.split("_")[1])
    text = str(neg_files_path[i])
    pro_name.append(number.split("_")[0])
    lig_name.append(number.split("_")[1])

    # Concat positive sample
    X_pos = np.loadtxt(pos_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_pos = X_pos[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_pos), axis=0)
    # Concat negative sample
    X_neg = np.loadtxt(neg_files_path[i], delimiter=',', dtype=np.float64, ndmin=2)
    X_neg = X_neg[np.newaxis,:,:,np.newaxis]
    X = np.concatenate((X,X_neg), axis=0)

print('X.shape:', X.shape)


# In[24]:


X = X.reshape(6000,-1)
print('X.shape:', X.shape)


# In[25]:


pro_name =list()
lig_name = list()
for i in range(0,num_pairs):
    text = str(pos_files_path[i])
    number =text[19:28]
    pro_name.append(number.split("_")[0])
    lig_name.append(number.split("_")[1])
    text2 = str(neg_files_path[i])
    number2 =text2[19:28]
    pro_name.append(number2.split("_")[0])
    lig_name.append(number2.split("_")[1])


# In[26]:


# Create Y: Dock-(1,0), No dock-(0,1)
Y = np.tile((1,0,0,1), num_pairs).astype(np.float64).reshape(-1,2)
print('Y.shape:', Y.shape)


# In[27]:


from sklearn.model_selection import train_test_split

indices = np.array(range(X.shape[0]))
X_train, X_test, Y_train, Y_test,pro_name_train,pro_name_test,lig_name_train,lig_name_test = train_test_split(X, Y, pro_name,lig_name,test_size=0.20, random_state=111)


# In[28]:


print(X_train.shape)
print(X_test.shape)


# In[84]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0, max_depth=25)
tree.fit(X_train, Y_train)


# In[85]:


# accuracy
print('train: {:.3f}'.format(tree.score(X_train, Y_train)))
print('test : {:.3f}'.format(tree.score(X_test, Y_test)))


# ## Xgboost

# In[63]:


Y2 = np.tile((1,0), num_pairs).astype(np.float64)
print('Y.shape:', Y2.shape)
print('X.shape:', X.shape)


# In[64]:


from sklearn.model_selection import train_test_split

indices = np.array(range(X.shape[0]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y2,test_size=0.20, random_state=111)


# In[65]:


import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(X_train, Y_train)


# In[66]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
#R^2
from sklearn.metrics import r2_score
print("R^2 train:%.3f, test:%.3f"%(r2_score(Y_train,y_train_pred),r2_score(Y_test, y_test_pred)))


# ## Lightgbm

# In[59]:


# Create Y: Dock-(1,0), No dock-(0,1)
Y2 = np.tile((1,0), num_pairs).astype(np.float64)
print('Y.shape:', Y2.shape)
print(Y2)


# In[60]:


print('X.shape:', X.shape)


# In[61]:


from sklearn.model_selection import train_test_split

indices = np.array(range(X.shape[0]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y2,test_size=0.20, random_state=111)


# In[70]:


import lightgbm as lgb

def get_lgb_clf():   
    model = lgb.LGBMClassifier(bagging_fraction=0.8,feature_fraction =  0.9,max_bin = 1000, learning_rate = 0.5,num_leaves = 1000,n_estimatiors =100)
    clf = model.fit(X_train, Y_train)
    return clf

lb = get_lgb_clf()
lb.fit(X_train, Y_train)


# In[71]:


y_train_pred = lb.predict(X_train)
y_test_pred = lb.predict(X_test)
#R^2
from sklearn.metrics import r2_score
print("R^2 train:%.3f, test:%.3f"%(r2_score(Y_train,y_train_pred),r2_score(Y_test, y_test_pred)))

