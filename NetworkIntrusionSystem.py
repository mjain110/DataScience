#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas_profiling as pf
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cols=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land", "wrong_fragment","urgent","hot","num_failed_logins","logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations", "num_shells","num_access_files","num_outbound_cmds","is_host_login", "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate", "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate", "dst_host_diff_srv_rate","dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate", "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]
len(cols)


# In[3]:


data=pd.read_csv('Train.txt',header=None)

data.head()


data.info()



# In[4]:


test=pd.read_csv('Test.txt',header=None)

test.head()

test.columns=cols


# In[5]:


cols


# In[6]:


data.columns=cols

data.head()

data.shape

data.wrong_fragment.value_counts()


# In[7]:


#p=pf.ProfileReport(data)

#p.to_file('data.html')


# In[8]:


data.describe()


# In[9]:


data[data.attack!='normal']


# In[10]:


data.protocol_type.value_counts()


# In[11]:


data.service.value_counts()


# In[12]:


data.flag.value_counts()


# In[13]:


data.logged_in.value_counts()


# In[14]:


import statsmodels.formula.api as smf


# In[15]:


data.attack.value_counts()


# In[16]:


data['attack_flag']=np.where(data.attack=='normal',0,1)


data.attack_flag.value_counts()




# In[17]:


data.head()


# In[18]:


data.dtypes


# In[19]:


data.duration


# In[20]:


num=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['int64','float64','int32','float32']]
cat=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object','str']]


# In[21]:


cat


cat.append('is_guest_login')

cat.append('attack_flag')
cat.append('logged_in')


cat.append('root_shell')


# In[22]:


# Removing service columns because of high cardinality
cat.remove('service')


# In[23]:



num.remove('attack_flag')

#removing nu_outbound_cmds column because of the constant 0
num.remove('num_outbound_cmds')

num.remove('logged_in')

#removing is_host_login column because of the constant 0
num.remove('is_host_login')


num.remove('is_guest_login')

num.remove('root_shell')



len(num)


# In[24]:


num


# In[25]:


data.num_failed_logins.value_counts()


failedlogins=pd.crosstab(data.num_failed_logins,data.attack_flag)

failedlogins.plot(kind="bar", 
                 figsize=(8,8),stacked=True)

#pd.crosstab(data.logged_in,data.attack_flag)

#sns.barplot(x='attack_flag',y='num_failed_logins',data=data)


# In[26]:


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


# In[27]:


num_data=data[num]

cat_data=data[cat]

num_data.apply(var_summary)

#num_data.head()

#num_data.su_attempted.value_counts()


# In[28]:


def outliercapping(x):
    x=x.clip_upper(x.quantile(0.99))
    x=x.clip_lower(x.quantile(0.01))
    return x
    


# In[29]:


num_data=num_data.apply(outliercapping)


# In[30]:


num_data.apply(var_summary)


# In[31]:


cat_data.apply(cat_summary)


# In[32]:


def attackclass(x):
    d = {'back':'DoS', 'land':'DoS' , 'neptune':'DoS','pod':'DoS','smurf':'DoS','teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','worm':'DoS'
      ,'satan':'Probe','nmap':'Probe','ipsweep':'Probe', 'portsweep':'Probe','mscan':'Probe','saint':'Probe',
      'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L','multihop':'R2L','warezmaster':'R2L','warezclient':'R2L',
     'spy':'R2L','xlock':'R2L','xsnoop':'R2L','snmpguess':'R2L','snmpgetattack':'R2L','httptunnel':'R2L','sendmail':'R2L','named':'R2L',
         'buffer_overflow':'U2R' ,'loadmodule':'U2R','rootkit':'U2R','perl':'U2R','rootkit':'U2R','sqlattack':'U2R','xterm':'U2R','ps':'U2R',
    'normal':'Normal'}
    if x in d.keys():
        return d[x]
    else:
        return ''
    
    
cat_data.drop('attack',axis=1,inplace=True)
#cat_data['attack_class']=cat_data.attack.apply(attackclass)


# In[33]:


cat_data.head()


# In[34]:


#cor=num_data.corr()

#plt.figure(figsize=(30,15))

#sns.heatmap(cor, annot=True, cmap = 'viridis')


# In[35]:


num_data.num_access_files.value_counts()


# In[36]:




# dropping numeric columns after outlier treatement

num_data.drop(columns=['land','wrong_fragment','urgent','num_failed_logins','su_attempted','num_root','num_file_creations','num_shells',
                       'num_access_files'], axis=1, inplace=True)


c=num_data.corr()

c.to_csv('numeric_corr.csv')


# In[37]:


def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
    #col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[38]:


#for c_feature in categorical_features
cat_data_new = cat_data
for c_feature in cat_data.columns:
    cat_data_new[c_feature] = cat_data_new[c_feature].astype('category')
    cat_data_new = create_dummies(cat_data_new , c_feature )


# In[39]:


cat_data_new.head()


# In[40]:


final_data= pd.concat([num_data,cat_data_new],axis=1)
final_data.head()


# In[41]:


import statsmodels.formula.api as sm
from sklearn import metrics

somersd_df = pd.DataFrame()
for num_variable in final_data.columns.difference(['attack_flag_1']):
    logreg = sm.logit(formula = str('attack_flag_1 ~ ')+str(num_variable), data=final_data)
    result = logreg.fit()
    y_score = pd.DataFrame(result.predict())
    y_score.columns = ['Score']
    somers_d = 2*metrics.roc_auc_score(final_data['attack_flag_1'],y_score) - 1
    temp = pd.DataFrame([num_variable,somers_d]).T
    temp.columns = ['Variable Name', 'SomersD']
    somersd_df = pd.concat([somersd_df, temp], axis=0)


# In[42]:


somersd_df.sort_values('SomersD',ascending=False)


# In[43]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = final_data[final_data.columns.difference(['attack_flag_1'])]
logreg = LogisticRegression()
rfe = RFE(logreg, 30)
rfe = rfe.fit(X, final_data[['attack_flag_1']] )
print(rfe.support_)
print(rfe.ranking_)


# In[44]:


X.columns


# In[45]:


logreg=LogisticRegression()

X=final_data[final_data.columns.difference(['attack_flag_1'])]
y=final_data[['attack_flag_1']]

result=logreg.fit(X,y)


result.score(X,y)
                


# In[46]:


test.head()

test['attack_flag']=np.where(test.attack=='normal',0,1)
    
cat.remove('attack')
test_data_num=test[num]

test_data_cat=test[cat]


# In[47]:


#for c_feature in categorical_features
test_final_data_new = test_data_cat
for c_feature in test_data_cat.columns:
    test_final_data_new[c_feature] = test_data_cat[c_feature].astype('category')
    test_final_data_new = create_dummies(test_final_data_new , c_feature )


# In[48]:


test_data_num.drop(columns=['land','wrong_fragment','urgent','num_failed_logins','su_attempted','num_root','num_file_creations','num_shells',
                       'num_access_files'], axis=1, inplace=True)


# In[49]:


test_final_data=pd.concat([test_data_num,test_final_data_new],axis=1)


# In[50]:


test_final_data.head()


# In[51]:


final_data.head()


# In[53]:


len(test_final_data.columns)


# In[55]:


logreg=LogisticRegression()

X=test_final_data[test_final_data.columns.difference(['attack_flag_1'])]
y=test_final_data[['attack_flag_1']]

result=logreg.fit(X,y)


result.score(X,y)
                


# In[ ]:




