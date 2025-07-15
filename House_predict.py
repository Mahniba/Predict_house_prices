#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('conda create -n houseprice python=3.10 -y')



# In[8]:


get_ipython().system('conda install jupyter')



# In[10]:


import pandas as pd
import numpy as np
import torch
import pickle
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[22]:


df = pd.read_csv("house_prices.csv")
df.head()


# In[76]:


df = pd.get_dummies("house_prices.csv")


# In[72]:


df.columns


# In[44]:


# Drop rows with missing values (for simplicity)
df = df.select_dtypes(include=[np.number]).dropna()

# Features and target
X = df.drop(['Price (in rupees)'], axis=1)
y = df['Price (in rupees)']


# In[50]:


X


# In[56]:


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# In[60]:


#Here the are 3 layers, the input, hidden and output layers
model = nn.Sequential(
    nn.Linear(X_train_tensor.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)


# In[62]:


#Does actual value compare to predicted values?(MSE)
loss_fn = nn.MSELoss()
#The optimizer adjusts the model’s weights based on the loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[63]:


#goes through the data hundred times to learn it better
epochs = 100
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    #Compares the model's predictions (y_pred) to the actual house prices (y_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)

    optimizer.zero_grad()
    #Compute how to fix the weights 
    loss.backward()
    #Update the weights
    optimizer.step()
    #Every 10 epochs, print the progress.
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# In[66]:


model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_loss = loss_fn(test_preds, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")


# In[68]:


model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_loss = loss_fn(test_preds, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")


# In[70]:


plt.scatter(y_test_tensor.numpy(), test_preds.numpy(), alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()


# In[ ]:





# In[74]:


torch.save()


# In[ ]:




