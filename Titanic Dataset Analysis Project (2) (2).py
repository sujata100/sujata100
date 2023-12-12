#!/usr/bin/env python
# coding: utf-8

# # TITANIC DATASET ANALYSIS

# We are all acquainted with the Titanic, The Unsinkable Ship, which sailed its first and last voyage in 1912. Even though the Titanic was made not to sink, there weren't enough lifeboats for everyone. Hence, resulted in the death of 1502 out of 2224 passengers and crew.
# 
# The Titanic Dataset link is a dataset curated on the basis of the passengers on titanic, like their age, class, gender, etc to predict if they would have survived or not. While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

# ![MERGE%20IMAGE1.png](attachment:MERGE%20IMAGE1.png)

# 
# # IMPORT LIBRARIES

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# # LOAD DATASET

# In[2]:


#Load the dataset
titanic = pd.read_csv('titanic.csv')


# In[3]:


#Length of the Dataset
len(titanic)


# In[4]:


#Shape of the Dataset
titanic.shape


# Passenger ID = To identify unique passengers.
# survival = survival 0 - No, 1 - Yes.
# Pclass =  Ticket class 1 - 1st, 2 - 2nd, 3 - 3rd.
# Sex = Male,Female.
# Age = Age in Years.
# SibSp = Number of siblings/spouses aboard the titanic.
# Parch  = of Parents/children aboard the titanic.
# Ticket = Ticket number of passenger fare Cabin number.
# Fare = Amount paid for the ticket.
# Cabin = Cabin of residence.
# Embarked = Port of Embarkation C - Cherbourg, Q - Queenstown, S - Southampton.  

# In[5]:


#Inspect the first few rows of the dataset
titanic.head()


# In[6]:


#Set the index to the passengerId
titanic = titanic.set_index('PassengerId')


# In[7]:


#Check out the data summary
#Age, Cabin and Embarkked has missing data
titanic.head()


# In[8]:


titanic.tail()


# In[9]:


titanic.columns


# In[10]:


#Identify data types of the 11 columns, add the states to the datadict
data = pd.DataFrame(titanic.dtypes)
data


# In[11]:


#Identify missing values of the 11 columns, add the stats to the data
data['MissingVal'] = titanic.isnull().sum()
data


# In[12]:


#Identify the number of missing values, For object nunique will the number of levels
#Add the stats of the data 
data['NUnique'] = titanic.nunique()
data


# In[13]:


#Identify the count for each variable, add the stats to data
data['Count'] = titanic.count()
data


# In[14]:


#Rename the 0 column
data = data.rename(columns = {0:'DataType'})
data


# In[15]:


#Get discriptive statistics on number datatypes
titanic.describe()


# # DATA ANALYSIS

# Data analysis is the process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information

# In[16]:


#Find out how many passenger survived vs died using countplot 


# In[17]:


import seaborn as sns


# In[18]:


#Countplot of survived vs not survived: 0 = Not survived , 1 = Survived
sns.countplot(x='Survived', data = titanic)


# In[19]:


#Male vs Female Survival
sns.countplot(x='Survived',data=titanic, hue='Sex')


# In[20]:


#Check null values
titanic.isna()


# In[21]:


#Visualize null values
sns.heatmap(titanic.isna())


# In[22]:


#Find the % of null values in age column
(titanic['Age'].isna().sum()/len(titanic['Age']))*100


# In[23]:


#Find the % of null values in Cabin column
(titanic['Cabin'].isna().sum()/len(titanic['Cabin']))*100


# In[24]:


#Checking how many passengers were survived or not
titanic.Survived.value_counts(normalize=True)


# In[25]:


#Only 38% of the passengers were survived, where as a majority 61% of the passenger did not survived in the disaster.


# # DATA CLEANING

# In[26]:


#Fill null values
titanic['Age'].mean()


# In[27]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[28]:


#Verifying null values


# In[29]:


titanic['Age'].isna().sum()


# In[30]:


#Using heatmap
sns.heatmap(titanic.isna())


# In[31]:


#Drop cabin column
titanic.drop('Cabin',axis=1,inplace=True)


# In[32]:


titanic.head()


# In[33]:


#Check for the numeric column
titanic.info()


# In[34]:


titanic.dtypes


# In[35]:


#Covert sex column to numerical columns
gender=pd.get_dummies(titanic['Sex'],drop_first=True)


# In[36]:


titanic['Gender']=gender


# In[37]:


titanic.columns


# In[38]:


titanic.head()


# In[39]:


#Drop the columns which are not required
titanic.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[40]:


titanic.head()


# In[41]:


titanic.isnull().sum()


# # DATA VISUALIZATION

# In[42]:


#countplot

fig,ax = plt.subplots(2,3,figsize =(16,10))

sns.countplot(x ="Survived",data=titanic,ax=ax[0][0])
sns.countplot(x ="Pclass",data=titanic,ax=ax[0][1])
sns.countplot(x ="Gender",data=titanic,ax=ax[0][2])
sns.countplot(x ="SibSp",data=titanic,ax=ax[1][0])
sns.countplot(x ="Parch",data=titanic,ax=ax[1][1])
sns.distplot(titanic['Age'],kde=False,ax=ax[1][2])


# We can clearly see that male survial rates is around 20% where as female survial rate is about 75% which suggests that gender has a strong relationship with the survival rates.
# 
# There is also a clear relationship between Pclass and the survival by referring to first plot below. Passengers on Pclass1 had a better survial rate of approx 60% whereas passengers on pclass3 had the worst survial rate of approx 22%
# 
# There is also a marginal relationship between the fare and survial rate.

# In[43]:


figbi, axesbi = plt.subplots(2,3, figsize=(16,10))
titanic.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
titanic.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
titanic.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
titanic.groupby('Gender')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x='Survived',y='Age',data=titanic,ax=axesbi[1,1])
sns.boxplot(x='Survived',y='Fare',data=titanic,ax=axesbi[1,2])


# # What is an Outlier?

# An outlier is a data point in a data set that is distinct from all the observations. A data point that lies outside the overall distribution of the dataset

# # What are the criteria to identify an outlier? 

# 1.Data point that falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile.

# # What is the reason for an outlier to exsists in a dataset?

# 1.Variablity in the data
# 2.An experimental measurement error

# # What are the impacts of having outliers in a dataset?

# 1.It causes various problems during our statistical analysis
# 2.It may cause a significant impact on the mean and the standard deviation

# In[44]:


sns.boxplot(x='Fare',data=titanic)


# # Using the IQR Interquantile range 

# 75% - 25% values in a dataset

# # Steps

# 1.Arrange the data in increasing order
# 2.Calculate first(q1) and third quartile(q3)
# 3.Find interqartile range(q3 - q1)
# 4.Find lower Bound q1 * 1.5
# 5.Find upper bound q3 * 1.5
# Anyting that lies ouside of lower and upper bound is an outlier

# In[45]:


q1 = titanic['Fare'].quantile(0.25)
q3 = titanic['Fare'].quantile(0.75)
iqr = q3 - q1


# In[46]:


q1,q3,iqr


# In[47]:


upper_limit = q3 + (1.5 * iqr)
lower_limit = q1 - (1.5 * iqr)
lower_limit, upper_limit


# In[48]:


sns.boxplot(x='Fare',data=titanic)


# In[49]:


#Find outliers
titanic.loc[(titanic['Fare'] > upper_limit) | (titanic['Fare'] < lower_limit)]


# In[50]:


#Trimming - Delete the outlier dta
new_titanic = titanic.loc[(titanic['Fare'] < upper_limit) & (titanic['Fare'] > lower_limit)]
print('Before removing outliers:', len(titanic))
print('After removing outliers:', len(new_titanic))
print('outliers:', len(titanic)-len(new_titanic))


# In[51]:


sns.boxplot(x='Fare',data=new_titanic)


# In[52]:


#Capping - change the outlier values to upper (or) lower limit values
new_titanic = titanic.copy()
new_titanic.loc[(new_titanic['Fare'] > upper_limit), 'Fare'] = upper_limit
new_titanic.loc[(new_titanic['Fare'] < lower_limit), 'Fare'] = lower_limit


# In[53]:


sns.boxplot(x='Fare',data=new_titanic)


# In[54]:


#pie chart which will give us the count of male=1, female=0 survived or did not survived
plt.figure(figsize=(5,5))
titanic.Gender.value_counts().plot(kind='pie')


# In[55]:


titanic.Gender.value_counts()


# In[56]:


titanic.groupby('Gender').Survived.mean().plot(kind='bar')
print(titanic.groupby('Gender').Survived.value_counts())


# Though there were more male passengers as compared to female passengers, but still more females survived as compared to males

# In[57]:


#Analyse people who survived and not survived based on their class
not_survived_class = titanic['Pclass'][titanic['Survived']==0]
survived_class = titanic['Pclass'][titanic['Survived']==1]

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
not_survived_class.value_counts().plot(kind='pie', title='people who didn\'t survived')

plt.subplot(1,2,2)
survived_class.value_counts().plot(kind='pie', title='people who survived')

print('People who survived: \n',survived_class.value_counts())
print('People who didn\'t survived: \n',not_survived_class.value_counts())


# In[58]:


# Analyze the people who survived and not survived based on the ticket fare

not_survived_fare = titanic['Fare'][titanic['Survived']==0]
survived_fare = titanic['Fare'][titanic['Survived']==1]

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
not_survived_fare.plot(kind='hist',title = 'People who didn\'t survive')


plt.subplot(1,2,2)
survived_fare.plot(kind='hist', title= 'People who survived')

print('Mean ticket fare of people who didn\'t survived: ', not_survived_fare.mean())
print('Mean ticket fare of people who survived: ', survived_fare.mean())


# # Heatmap

# A heat map is a two-dimensional representation of data in which various values are represented by colors. A simple heat map provides an immediate visual summary of information across two axes, allowing users to quickly grasp the most important or relevant data points. More elaborate heat maps allow the viewer to understand complex data sets.

# In[59]:


sns.heatmap(titanic.corr(), cmap="tab20", annot= True)
plt.show()


# In[60]:


titanic.corr()


# ## Separate Depandent and  Independent variables

# In[62]:


x = titanic[['Pclass','Age','SibSp','Parch','Fare','Gender']]
y = titanic['Survived']


# In[63]:


x


# In[64]:


y


# # DATA MODELING

# Splitting the Dataset

# In[65]:


#Import train test split method
from sklearn.model_selection import train_test_split


# In[66]:


#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,  train_size=0.7, test_size=0.3, random_state=101)


# In[67]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # LOGISTIC REGRESSION

# Logistic regression is a data analysis technique that uses mathematics to find the relationships between two data factors. It then uses this relationship to predict the value of one of those factors based on the other. The prediction usually has a finite number of outcomes, like yes or no.

# In[68]:


from sklearn.linear_model import LogisticRegression


# In[69]:


#Fit Logistic Regression


# In[70]:


lr = LogisticRegression()


# In[71]:


lr.fit(x_train,y_train)


# In[72]:


#Predict


# In[73]:


predict = lr.predict(x_test)


# In[74]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cf = confusion_matrix(predict,y_test)
cf


# In[75]:


ac=cf.diagonal().sum()/cf.sum()*100
ac


# In[76]:


print(classification_report(predict,y_test))


# # DECISION TREE

# A decision tree is one of the most powerful tools of supervised learning algorithms used for both classification and regression tasks. It builds a flowchart-like tree structure where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. It is constructed by recursively splitting the training data into subsets based on the values of the attributes until a stopping criterion is met, such as the maximum depth of the tree or the minimum number of samples required to split a node

# In[77]:


from sklearn.tree import DecisionTreeClassifier 
dt  = DecisionTreeClassifier()


# In[78]:


dt.fit(x_train,y_train)


# In[79]:


pre_x1 = dt.predict(x_test)
pre_x1


# In[80]:


cf1 = confusion_matrix(pre_x1,y_test)
cf1


# In[81]:


acc1 = cf1.diagonal().sum()/cf1.sum()*100
acc1


# In[82]:


print(classification_report(pre_x1,y_test))


# # RANDOM FOREST

# Random Forest Algorithm widespread popularity stems from its user-friendly nature and adaptability, enabling it to tackle both classification and regression problems effectively. The algorithm’s strength lies in its ability to handle complex datasets and mitigate overfitting, making it a valuable tool for various predictive tasks in machine learning.

# In[83]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[84]:


rf.fit(x_train,y_train)


# In[85]:


pre_x2 = rf.predict(x_test)
pre_x2


# In[86]:


cf2 = confusion_matrix(pre_x2,y_test)
cf2


# In[87]:


acc2 = cf2.diagonal().sum()/cf2.sum()*100
acc2


# In[88]:


print(classification_report(pre_x2,y_test))


# In[89]:


#MODEL EVALUATION
models = pd.DataFrame({'Model':['Logistic Regression','Decision Tree','Random Forest'], 'Accuracy_Score':[ac,acc1,acc2]})
models.sort_values(by='Accuracy_Score',ascending=False)


# In[90]:


c = ['Purple', 'Yellow', 'Maroon']
plt.bar(models["Model"],models["Accuracy_Score"],color = c)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of all Models')
plt.show()


# From the above analysis we can see that Random Forest give us the higher accuracy which is 83.20 % , while Logistic regression and Decision Tree give us the less accuracy.So we can say Random Forest is the best model for the analysis.

# In[ ]:




