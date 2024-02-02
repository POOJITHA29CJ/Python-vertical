#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd


# In[24]:


df=pd.read_csv('hiring.csv')
print(df.to_string())


# In[25]:


print(df.isna().sum())


# In[26]:


x=int(df['test_score(out of 10)'].mean())
print(x)


# In[27]:


df['test_score(out of 10)'].fillna(x,inplace=True)
print(df.to_string())
print(df['test_score(out of 10)'])


# In[31]:


df['experience'].fillna('zero',inplace=True)
print(df.to_string())
print(df['experience'])


# In[32]:


print(df.isna().sum())


# In[33]:


from word2number import w2n
df['experience']=df['experience'].apply(w2n.word_to_num)
df


# In[35]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[37]:


x=df.drop(columns=['salary($)'])
y=df['salary($)']


# In[38]:


print(x.to_string())
print(y.to_string())


# In[40]:


reg.fit(x,y)


# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[42]:


print(x_train.shape)
print(x_test.shape)


# In[43]:


n=int(input("Enter how many data"))
lists=[]
for i in range(n):
    x=list(map(int,input("Experience,TestScore,InterviewScore : ").split()))
    lists.append(x)
ans=reg.predict(lists)
for i in ans:
    print(i)


# In[45]:


n=int(input("Enter how many data: "))
lists=[]
for i in range(n):
    a=int(input(f"Enter experience of {i+1} person: "))
    b=int(input(f"Enter test score of {i+1} person:"))
    c=int(input(f"Enter interview score of {i+1} person:"))
    nested=list([a,b,c])
    lists.append(nested)
ans=reg.predict(lists)
for i in ans:
    print(f"Salary - {i} ")
    


# In[ ]:




