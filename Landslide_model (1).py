#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np


# In[43]:


df=pd.read_csv(r"E:/NASA/catalog.csv",index_col='id')


# In[44]:


df.head()


# In[45]:


df.columns


# In[46]:


df.isna().sum()


# In[47]:


df.drop(columns=['location_description','source_link','storm_name','continent_code','country_code','injuries','source_name'
                ,'landslide_size','fatalities','geolocation','date','time','country_name','hazard_type','city/town'
                ,'state/province','trigger'],inplace=True)


# In[48]:


df.isna().sum()


# In[49]:


df.distance.fillna(2.54860,inplace=True)


# In[50]:


df.landslide_type.fillna('Landslide',inplace=True)


# In[51]:


df.longitude.fillna('-77.2682',inplace=True)


# In[52]:


df.latitude.fillna('38.6009',inplace=True)


# In[53]:


df.isna().sum()


# In[54]:


df.landslide_type.replace('Landslide',0,inplace=True)


# In[55]:


df.landslide_type.replace('Mudslide',1,inplace=True)


# In[56]:


df.landslide_type.replace('Rockfall',2,inplace=True)


# In[57]:


df.landslide_type.replace('Complex',3,inplace=True)


# In[58]:


df.landslide_type.replace('Other',4,inplace=True)


# In[59]:


df.landslide_type.replace('Riverbank collapse',5,inplace=True)


# In[60]:


df.landslide_type.replace('Debris flow',6,inplace=True)


# In[61]:


df.landslide_type.replace('Creep',7,inplace=True)


# In[62]:


df.landslide_type.replace('mudslide',8,inplace=True)


# In[63]:


df.landslide_type.replace('Snow avalanche',9,inplace=True)


# In[64]:


df.landslide_type.replace('Lahar',10,inplace=True)


# In[65]:


df.landslide_type.replace('Rockslide',11,inplace=True)


# In[66]:


df.landslide_type.replace('Unknown',12,inplace=True)


# In[67]:


df.landslide_type.replace('landslide',0,inplace=True)
df.landslide_type.value_counts()


# In[68]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[69]:


y = df.landslide_type


# In[70]:


df.columns


# In[71]:


features = ['population', 'distance', 'latitude', 'longitude']
X = df[features]


# In[72]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[73]:


iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)


# In[74]:


# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))


# In[75]:


# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# In[76]:


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# In[77]:


rf_model_on_full_data  = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(train_X, train_y)


# In[78]:


test_preds = rf_model_on_full_data.predict(val_X)


# In[79]:


test_preds


# In[80]:


test_example = pd.DataFrame(data={
    'population': [10000],
    'distance': [20], 
    'latitude': [40.5175], 
    'longitude': [-81.4305]
})


# In[81]:


m=rf_model_on_full_data.predict(test_example)


# In[82]:


round(m[0])


# In[ ]:





# In[ ]:





# In[ ]:




