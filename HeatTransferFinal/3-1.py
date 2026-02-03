#!/usr/bin/env python
# coding: utf-8

# # 第三題b(i)

# In[1]:



temp = [[85, 85, 85, 85, 20]]

fo = 0.5
for p in range (1,10):
    newtemp = list()
    newtemp.append(round(0.5*(2*temp[p-1][1]),1))
    for i in range(1,4):
        #print(i)
        newtemp.append(round(0.5*(temp[p-1][i-1]+temp[p-1][i+1]),1))
    newtemp.append(20)
    temp.append(newtemp)

p = 0
for i in temp:
    print(p,i)
    p += 1


# ## ans = [61.7, 55.6, 49.5, 34.8, 20]

# In[ ]:



    


# In[ ]:




