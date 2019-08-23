#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas_profiling as pf
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


cols=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land", "wrong_fragment","urgent","hot","num_failed_logins","logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations", "num_shells","num_access_files","num_outbound_cmds","is_host_login", "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate", "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate", "dst_host_diff_srv_rate","dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate", "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]
len(cols)


# In[3]:


train=pd.read_csv('Train.txt',header=None)

train.head()


train.shape


# In[4]:



test=pd.read_csv('Test.txt',header=None)

test.shape

test.tail()


# In[5]:


train.columns=cols

train.head()


# In[6]:


#pf.ProfileReport(train)


# In[ ]:





# In[7]:


train.info()


# In[8]:


train.protocol_type.value_counts()


# In[9]:


train.service.value_counts()


# In[10]:


train.logged_in.value_counts()


# In[11]:


train.attack.value_counts()


# In[12]:


train['attack_flag']=np.where(train.attack=='normal',0,1)

train.drop('attack',axis=1,inplace=True)


train.head()


# In[13]:


import statsmodels.formula.api as smf


# In[14]:


train.num_outbound_cmds.value_counts()


# In[15]:


cat=['protocol_type','service','flag','land','logged_in','root_shell','su_attempted','is_host_login','is_guest_login','attack_flag']

num=[]
for c in train.columns.difference(cat):
    num.append(c)

num


# dropping num_outbound_cmds because of constant 0 value

num.remove('num_outbound_cmds')

# dropping because of >99.9 zeroes..

num.remove('hot')
num.remove('num_access_files')
num.remove('num_compromised')
num.remove('num_failed_logins')
num.remove('num_file_creations')
num.remove('num_root')


#dropping the variables because of high correlation with the other variable


num.remove('dst_host_srv_serror_rate')
num.remove('dst_host_srv_rerror_rate')
num.remove('srv_rerror_rate')
num.remove('srv_serror_rate')


#dropping because all the values are zeros after outlier's treatment

num.remove('num_shells')
num.remove('urgent')
num.remove('wrong_fragment')


# In[16]:


# Creating Data audit Report
# Use a general function that returns multiple values
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(),  x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])


# Creating Data audit Report
# Use a general function that returns multiple values
def cat_summary(x):
    return pd.Series([x.count(), x.isnull().sum()], 
                  index=['N', 'NMISS'])


# In[17]:


train_num=train[num]

#train_num.apply(var_summary).to_csv('num.csv')

train_num.apply(var_summary)



# In[18]:


train_cat=train[cat]

train_cat.head()

train_cat.apply(cat_summary)


# In[19]:


train_num.head()

#num_data.num_compromised.unique()


# In[20]:


#train_num.corr().to_csv('correlation_before.csv')

train_num.corr()


# In[21]:


def outliercapping(x):
    x=x.clip_upper(x.quantile(0.99))
    x=x.clip_lower(x.quantile(0.01))
    return x


    


# In[22]:


train_num=train_num.apply(outliercapping)

train_num.apply(var_summary).to_csv('summary_after.csv')

summary=train_num.apply(var_summary)


# In[23]:


from scipy import stats


chi2_score = []
p_val = []

y = train_cat.attack_flag
for col in cat:
    xtab = pd.crosstab(train_cat[col],y,margins=True)
    ch2 = stats.chi2_contingency(observed=xtab)
    chi2_score.append(ch2[0])
    p_val.append(ch2[1])
    
cat_vars = pd.Series(cat,name="Column")
chi2_score = pd.Series(chi2_score,name="chi2_score")
p_val = pd.Series(p_val,name = "p_value")
chi2=pd.concat([cat_vars,chi2_score,p_val],axis=1)

chi2.sort_values('chi2_score',ascending=False)

chi2[chi2.p_value<0.05]

#p value >0.05 we accept null hypothesis which means those variables has no impact
#chi2[chi2.p_value>0.05]

# land and is_host_login has no impact on the target variable


# In[24]:


cat.remove('land')
cat.remove('is_host_login')

cat.remove('service')


# In[25]:


train_cat=train_cat[cat]


# In[26]:


# An utility function to create dummy variable
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[27]:


for c_feature in cat:
    train_cat = create_dummies( train_cat, c_feature )


# In[28]:


train_cat.head()


# In[29]:


train_final_data=pd.concat([train_num,train_cat],axis=1)

train_final_data.head()


# In[30]:


import statsmodels.formula.api as sm
from sklearn import metrics

somersd_df = pd.DataFrame()
for num_variable in train_final_data.columns.difference(['attack_flag_1']):
    logreg = sm.logit(formula = str('attack_flag_1 ~ ')+str(num_variable), data=train_final_data)
    result = logreg.fit()
    y_score = pd.DataFrame(result.predict())
    y_score.columns = ['Score']
    somers_d = 2*metrics.roc_auc_score(train_final_data['attack_flag_1'],y_score) - 1
    temp = pd.DataFrame([num_variable,somers_d]).T
    temp.columns = ['Variable Name', 'SomersD']
    somersd_df = pd.concat([somersd_df, temp], axis=0)
    
somersd_df.sort_values('SomersD',ascending=False)


# In[31]:


features="+".join(train_final_data.columns.difference(['attack_flag_1']))

features


# In[32]:


from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

a,b = dmatrices(formula_like='attack_flag_1 ~ '+ 'count+diff_srv_rate+dst_bytes+dst_host_count+dst_host_diff_srv_rate+dst_host_rerror_rate+dst_host_same_src_port_rate+dst_host_same_srv_rate+dst_host_serror_rate+dst_host_srv_count+dst_host_srv_diff_host_rate+duration+flag_REJ+flag_RSTO+flag_RSTOS0+flag_RSTR+flag_S0+flag_S1+flag_S2+flag_S3+flag_SF+flag_SH+is_guest_login_1+last_flag+logged_in_1+protocol_type_tcp+protocol_type_udp+rerror_rate+root_shell_1+same_srv_rate+serror_rate+src_bytes+srv_count+srv_diff_host_rate+su_attempted_1+su_attempted_2', data = train_final_data, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif["features"] = b.columns

vif.head()


# In[33]:


vif


# In[34]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = train_final_data[train_final_data.columns.difference(['attack_flag_1'])]
logreg = LogisticRegression()
rfe = RFE(logreg, 30)
rfe = rfe.fit(X, train_final_data[['attack_flag_1']] )
print(rfe.support_)
print(rfe.ranking_)


# In[35]:


RFEFeatures=X.columns[rfe.support_]

RFEFeatures


# In[36]:


from sklearn.model_selection import train_test_split

#X=train_final_data[RFEFeatures]

X=train_final_data[train_final_data.columns.difference(['attack_flag_1'])]

y=train_final_data['attack_flag_1']

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=123)


# In[37]:


from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

result=logreg.fit(train_X,train_y)


result.score(train_X,train_y)


# In[38]:


import sklearn.metrics as metrics

df=pd.DataFrame(result.predict(test_X))

df.columns=['pred']


actual_y=test_y

actual_y=actual_y.reset_index(drop=True)

actual_y

d=pd.concat([df,actual_y],axis=1)

d.columns=['pred','actual']

print(metrics.accuracy_score(d.actual,d.pred))


# In[39]:


test.head()

test.columns=cols

test_num=test[num]

test_num.apply(var_summary).to_csv('test_summary.csv')


# In[40]:


df=summary[(summary.index=='MAX')| (summary.index=='MIN')]
s=df.iloc[-1:].T
p=df.iloc[-2:].T



lmax=s['MAX'].tolist()
lmin=p['MIN'].tolist()

lmax




 


# In[44]:


def testoutliertreatment(x):
    x=x.clip_upper(lmax[2])
    x=x.clip_lower(lmin[2])
    return x

   


d=pd.DataFrame(test_num['dst_bytes'])

s=d.apply(testoutliertreatment)

s



# In[ ]:




