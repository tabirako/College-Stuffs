#!/usr/bin/env python
# coding: utf-8

# ## 第三題b(ii)

# In[1]:


import matplotlib.pyplot as plt


# ## Δt = 300s

# In[2]:



temp = [[85, 85, 85, 85, 20]]


for p in range (1,91):
    newtemp = list()
    newtemp.append(round(0.5*(2*temp[p-1][1]),1))
    for i in range(1,4):
        #print(i)
        newtemp.append(round(0.5*(temp[p-1][i-1]+temp[p-1][i+1]),1))
    newtemp.append(20)
    temp.append(newtemp)
    
    """
    diff = 0
    for i in range(5):
        if abs(temp[p][i]-temp[p-1][i]) >= 0.01:
            continue
        else:
            diff += 1
            
    if(diff != 5):
        break
    """
    
    
p = 0
for i in temp:
    print(p,i)
    p += 1


# ## Δt = 300s

# In[3]:


left = []
mid = []
right = []
for i in range(len(temp)):
    left.append(temp[i][0])
    mid.append(temp[i][2])
    right.append(temp[i][4])
    
plt.plot(left)
plt.plot(mid)
plt.plot(right)

plt.show()
    

    


# In[8]:



temp = [[85, 85, 85, 85, 20]]

fo = 0.125

def calc2(fo,temp):
    for p in range (1,200):
        newtemp = list()
        newtemp.append(round(fo*(2*temp[p-1][1])+(1-2*fo)*temp[p-1][1],1))
        for i in range(1,4):
            #print(i)
            newtemp.append(round(fo*(temp[p-1][i-1]+temp[p-1][i+1])+(1-2*fo)*temp[p-1][i],1))
        newtemp.append(20)
        temp.append(newtemp)
        
        """
        diff = 0
        for i in range(5):
            if (temp[p-1][i]-temp[p][i]) < 1:
                diff += 1
            
        if diff == 5:
            return temp
        """
    return temp

temp = calc2(fo,temp)
    
p = 0
for i in range(len(temp)):
    print(p,temp[i])
    p += 1


# In[5]:


left = []
mid = []
right = []
for i in range(len(temp)):
    left.append(temp[i][0])
    mid.append(temp[i][2])
    right.append(temp[i][4])
    
plt.plot(left)
plt.plot(mid)
plt.plot(right)

plt.show()
    


# In[ ]:




