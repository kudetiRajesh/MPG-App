#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[3]:


df=pd.read_csv(r"C:\Users\admin\Desktop\auto p\Auto MPG Reg.csv")


# In[4]:


df


# In[5]:


df.horsepower=pd.to_numeric(df.horsepower,errors='coerce')


# In[6]:


df.horsepower=df.horsepower.fillna(df.horsepower.median())


# In[7]:


#split data 
y=df.mpg
X=df.drop(['carname','mpg'],axis=1)


# In[8]:


#define multiple models as a dictinary 
model={'Linear regression':LinearRegression(),"Decision Tree":DecisionTreeRegressor(),"Random Forest":RandomForestRegressor()
       ,"grandianboosting":GradientBoostingRegressor()}


# In[9]:


#sidebar for model selection
selected_model=st.sidebar.selectbox("select a ml model",list(model.keys()))


# In[10]:


#ml model selection parameters 
if selected_model=='Linear regression':
    model=LinearRegression()
elif selcted_model=="Decision Tree":
    max_depth=st.sidebar.slider("max_depth",8,16,2)
    model=DecisionTreeRegressor( max_depth= max_depth)
elif selected_mdoel=="Random Forest":
    n_estimators=st.sidebar.slider("num of trees",100,500,50)
    model=RandomForestRegressor(n_estimators=n_estimators)
elif selected_model=="grandianboosting":
    
    n_estimators=st.sidebar.slider("num of trees",100,500,50)
    model=GradientBoostingRegressor(n_estimators=n_estimators)


# In[11]:


#train the  model
model.fit(X,y)



# In[12]:


#define application page parameters
st.title("Predict mileage per gallon")
st.markdown("Model to predict Milleage of Car")
st.header("Car Features")

col1,col2,col3,col4=st.columns(4)
with col1:
    cylinders=st.slider("Cylinders",2,8,1)
    displacement=st.slider("Displacement",50,500,10)
with col2:
    horsepower=st.slider("horsepower",50,500,10)
    weight=st.slider('weight',1500,6000,250)
with col3:
    acceleration=st.slider('accel',8,25,1)
    modelyear=st.slider('year',70,85,1)
with col4:
    origin=st.slider("origin",1,3,1)


# In[13]:


#mdoel predictions
rsquare=model.score(X,y)


# In[15]:


y_pred= model.predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,origin]]))


# In[ ]:


#display results
st.header("ml model result")
st.write(f"selected model:{selected_model}" )
st.write(f"Rsquare:{rsquare}")
st.write(f"predict:{y_pred}")

