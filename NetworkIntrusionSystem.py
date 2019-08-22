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


train=pd.read_csv('Train.txt',header=None)

train.shape


# In[4]:


test=pd.read_csv('Test.txt',header=None)

test.shape


# In[5]:


data=train.append(test)

data.shape


# In[6]:


data.columns=cols

data.head()



# In[7]:


#p=pf.ProfileReport(data)

#p.to_file('data.html')


# In[8]:


data.describe()


# In[9]:


data.head()


# In[10]:


data.dtypes


# In[11]:


num=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['int64','float64','int32','float32']]
cat=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object','str']]


# In[12]:


cat

cat.append('is_guest_login')

cat.append('logged_in')

cat.append('root_shell')

cat


# In[13]:


# Removing service columns because of high cardinality
cat.remove('service')


# In[14]:



#removing nu_outbound_cmds column because of the constant 0
num.remove('num_outbound_cmds')

num.remove('logged_in')

#removing is_host_login column because of the constant 0
num.remove('is_host_login')

num.remove('is_guest_login')

num.remove('root_shell')


# dropping numeric columns after outlier treatement
# all these variables have zero >99%

num.remove('land')
num.remove('wrong_fragment')
num.remove('urgent')
num.remove('num_failed_logins')
num.remove('su_attempted')
num.remove('num_root')
num.remove('num_file_creations')
num.remove('num_shells')
num.remove('num_access_files')


#dropping the variables because of high correlation with the other variable


num.remove('dst_host_srv_serror_rate')
num.remove('dst_host_srv_rerror_rate')
num.remove('srv_rerror_rate')
num.remove('srv_serror_rate')


num


# In[15]:


#deriving dependent variable attack_flag from the attack variable

data['attack_flag']=np.where(data.attack=='normal',0,1)

data.attack_flag.value_counts()


# In[16]:


cat.append('attack_flag')


# In[17]:


num


# In[18]:


train_final_data=data.iloc[0:len(train)]

test_final_data=data.iloc[len(train):len(data)]

print(test_final_data.shape)
print(train_final_data.shape)


                


# In[19]:


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


# In[20]:


num_data=train_final_data[num]

cat_data=train_final_data[cat]


num_data_test=test_final_data[num]
cat_data_test=test_final_data[cat]

num_data.apply(var_summary)

#num_data.head()

#num_data.su_attempted.value_counts()


# In[21]:


def outliercapping(x):
    x=x.clip_upper(x.quantile(0.99))
    x=x.clip_lower(x.quantile(0.01))
    return x
    


# In[22]:


num_data=num_data.apply(outliercapping)


# In[23]:


num_data.apply(var_summary)


# In[24]:


cat_data.apply(cat_summary)


# In[25]:


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
    
cat_data_test.drop('attack',axis=1,inplace=True)

#cat_data['attack_class']=cat_data.attack.apply(attackclass)


# In[26]:


cat_data.head()


# In[27]:


#cor=num_data.corr()

#plt.figure(figsize=(30,15))

#sns.heatmap(cor, annot=True, cmap = 'viridis')


# In[28]:


c=num_data.corr()

c.to_csv('numeric_corr.csv')


# In[29]:


def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
    #col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[30]:


#for c_feature in categorical_features
cat_data_new = cat_data
for c_feature in cat_data.columns:
    cat_data_new[c_feature] = cat_data_new[c_feature].astype('category')
    cat_data_new = create_dummies(cat_data_new , c_feature )
    

cat_data_test_new=cat_data_test
for c_feature in cat_data_test.columns:
    cat_data_test_new[c_feature] = cat_data_test_new[c_feature].astype('category')
    cat_data_test_new = create_dummies(cat_data_test_new , c_feature )


# In[31]:


cat_data_new.head()


# In[32]:


train_final_data=pd.concat([num_data,cat_data_new],axis=1)

test_final_data=pd.concat([num_data_test,cat_data_test_new],axis=1)

print(train_final_data.shape)
print(test_final_data.shape)


# In[33]:


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


# In[34]:


somersd_df.sort_values('SomersD',ascending=False)


# In[35]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = train_final_data[train_final_data.columns.difference(['attack_flag_1'])]
logreg = LogisticRegression()
rfe = RFE(logreg, 30)
rfe = rfe.fit(X, train_final_data[['attack_flag_1']] )
print(rfe.support_)
print(rfe.ranking_)


# In[36]:


RFEFeatures=X.columns[rfe.support_]

RFEFeatures


# In[37]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


# In[38]:


X=train_final_data[train_final_data.columns.difference(['attack_flag_1'])]
y=train_final_data[['attack_flag_1']]

kbest=SelectKBest(f_classif,k=30).fit(X,y)


# In[39]:


kbest.get_support()


# In[40]:


kbestfeatures=X.columns[kbest.get_support()]

kbestfeatures


# In[41]:


logreg=LogisticRegression()

X=train_final_data[train_final_data.columns.difference(['attack_flag_1'])]
y=train_final_data[['attack_flag_1']]

result=logreg.fit(X,y)


result.score(X,y)
                


# In[42]:


from sklearn.model_selection import train_test_split


y=train_final_data['attack_flag_1']

#X=train_final_data[kbestfeatures]
X=train_final_data[train_final_data.columns.difference(['attack_flag_1'])]

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=123)


# In[43]:


import statsmodels.formula.api as smf
import sklearn.metrics as metrics

s=''
for c in kbestfeatures:
    s=s+ c + '+'

print(s)


f='attack_flag_1 ~ count+diff_srv_rate+dst_bytes+dst_host_count+dst_host_diff_srv_rate+dst_host_rerror_rate+dst_host_same_src_port_rate+dst_host_same_srv_rate+dst_host_serror_rate+dst_host_srv_count+dst_host_srv_diff_host_rate+flag_REJ+flag_RSTO+flag_RSTOS0+flag_RSTR+flag_S0+flag_S1+flag_S2+flag_SF+flag_SH+is_guest_login_1+last_flag+logged_in_1+num_compromised+protocol_type_tcp+protocol_type_udp+rerror_rate+same_srv_rate+serror_rate+srv_diff_host_rate'
r=smf.logit(formula=f,data=train_final_data)

result=r.fit()


print(result.summary2())
    


# In[44]:


from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

results=logreg.fit(train_X,train_y)
results.score(train_X,train_y)
                
#results.coef_

#results.score



# In[45]:



#Predicting the test cases
test_pred = pd.DataFrame( { 'actual':  test_y,
                           'predicted': results.predict( test_X ) } )

test_pred.head(20)


# In[46]:


test_final_X=test_final_data[test_final_data.columns.difference(['attack_flag_1'])]
test_final_y=test_final_data['attack_flag_1']

#Predicting the test cases
test_final_pred = pd.DataFrame( { 'actual':  test_final_y,
                           'predicted': results.predict( test_final_X ) } )

test_final_pred.head(10)

#print(test_final_data.shape)


# In[47]:


# Creating a confusion matrix for the train->test

from sklearn import metrics

cm = metrics.confusion_matrix( test_pred.actual,

                            test_pred.predicted, [1,0] )
cm


# In[48]:


# Creating a confusion matrix for the actual test data

from sklearn import metrics

cm = metrics.confusion_matrix( test_final_pred.actual,
                            test_final_pred.predicted, [1,0] )
cm


# In[49]:


#train->test score

score = metrics.accuracy_score( test_pred.actual, test_pred.predicted )
round( float(score), 2 )


# In[50]:


#Acutal test data score

score = metrics.accuracy_score( test_final_pred.actual, test_final_pred.predicted )
round( float(score), 2 )


# In[51]:


train_gini = 2*metrics.roc_auc_score(train_y, results.predict(train_X)) - 1
print("The Gini Index for the model built on the Train Data is : ", train_gini)

test_gini = 2*metrics.roc_auc_score(test_y, results.predict(test_X)) - 1
print("The Gini Index for the model built on the Test Data is : ", test_gini)

train_auc = metrics.roc_auc_score(train_y, results.predict(train_X))
test_auc = metrics.roc_auc_score(test_y, results.predict(test_X))

print("The AUC for the model built on the Train Data is : ", train_auc)
print("The AUC for the model built on the Test Data is : ", test_auc)
                                 


# In[52]:


pred_y=results.predict(train_X)


print(metrics.classification_report(train_y,pred_y ))


# In[53]:


pred_y=results.predict(test_final_X)


print(metrics.classification_report(test_final_y,pred_y ))


# In[54]:


#test_final_data['Pred']=results.predict(test_final_X)

#test_final_data.head()


# In[55]:


from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(max_depth=3)
    
dtree=dtree.fit(train_X,train_y)


# In[56]:


pred = pd.DataFrame(dtree.predict( train_X ), index=train_X.index)


d=pd.concat([train_y,pred],axis=1)

d.head()

d.columns=['actual','pred']


# In[57]:


print(metrics.accuracy_score( d.actual, d.pred ))


# In[58]:


test_pred=dtree.predict(test_final_X)

test_pred

pred=pd.DataFrame(test_pred)


new=pd.concat([test_final_y,pred],axis=1)

new.columns=['actual','predicted']

print(metrics.accuracy_score( new.actual,  new.predicted ))


# In[59]:


# classifying attackclass



train_final_data.shape



# In[60]:


train.columns=cols

test.columns=cols

train_final_data['attack']=train.attack
test_final_data['attack']=test.attack

test_final_data.head()


# In[61]:


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
    

        


# In[62]:


train_final_data['attack_class'] = train_final_data.attack.apply(attackclass)

test_final_data['attack_class'] = test_final_data.attack.apply(attackclass)

test_final_data.head()


# In[63]:


test_final_data.attack_class.unique()


# In[64]:


from sklearn.preprocessing import LabelEncoder

enc= LabelEncoder()

train_final_data['class_flag']=enc.fit_transform(train_final_data.attack_class)

test_final_data['class_flag']=enc.fit_transform(test_final_data.attack_class)

test_final_data.class_flag.unique()

test_final_data=test_final_data[test_final_data.attack_class!='']


# In[65]:


train_final_data.attack_class.unique()


# In[66]:


train_final_data.drop(['attack','attack_class','attack_flag_1'],axis=1,inplace=True)

test_final_data.drop(['attack','attack_class','attack_flag_1'],axis=1,inplace=True)

test_final_data.head()


# In[67]:


from sklearn.model_selection import train_test_split,GridSearchCV



y=train_final_data['class_flag']

#X=train_final_data[kbestfeatures]
X=train_final_data[train_final_data.columns.difference(['class_flag'])]

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=123)


# In[68]:


param_grid = {'max_depth': np.arange(3, 8),
             'max_features': np.arange(3,10)}


# In[69]:


tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5)
tree.fit( train_X, train_y )


# In[70]:


tree.best_score_


# In[71]:


tree.best_params_


# In[72]:


tree.best_estimator_


# In[73]:


pred_y=tree.predict(train_X)


# In[74]:


print(metrics.classification_report(train_y, pred_y))


# In[75]:


train_y.value_counts()


# In[76]:


actual_test_X=test_final_data[test_final_data.columns.difference(['class_flag'])]

actual_test_y=test_final_data['class_flag']

pred_y=tree.predict(actual_test_X)

print(metrics.accuracy_score(actual_test_y,pred_y))
#actual_test_X.head()
print(metrics.classification_report(actual_test_y, pred_y))


# In[77]:


from sklearn.ensemble import RandomForestClassifier


# In[79]:


radm_clf = RandomForestClassifier(oob_score=True,n_estimators=100 , max_features=6, n_jobs=-1)
radm_clf.fit( train_X, train_y )


# In[86]:


radm_clf.feature_importances_


# In[82]:


radm_test_pred = pd.DataFrame( { 'actual':  train_y,
                            'predicted': radm_clf.predict( train_X ) } )

metrics.accuracy_score( radm_test_pred.actual, radm_test_pred.predicted )

