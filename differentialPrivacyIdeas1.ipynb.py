#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


num_entries = 5000
db = torch.rand(num_entries) > 0.5
db


# In[3]:


db[0:5]


# In[4]:


remove_index = 2
torch.cat((db[0:2], db[3:]))[0:5]


# In[5]:


def get_parallel_db(db, remove_index):
    return torch.cat((db[0:remove_index], 
                      db[remove_index+1:]))


# In[6]:


parallel_dbs = list()


# In[7]:


def get_parallel_dbs(db):
    parallel_dbs = list()
    for i in range(len(db)):
        pdb = get_parallel_db(db, i)
        parallel_dbs.append(pdb)
    return parallel_dbs


# In[8]:


pdbs = get_parallel_dbs(db)


# In[9]:


def create_db_and_parallels(num_entries):
    db = torch.rand(num_entries) > 0.5
    pdbs = get_parallel_dbs(db)
    return db, pdbs


# In[10]:


db, pdbs = create_db_and_parallels(20)


# In[11]:


def query(db):
    return db.sum()


# In[12]:


def sensitivity(query, n_entries):
    db, pdbs = create_db_and_parallels(5000)
    full_db_result = query(db)
    max_distance = 0
    for pdb in pdbs:
        pdb_result = query(pdb)
        db_distance = torch.abs(pdb_result - full_db_result)    
        if db_distance > max_distance:
            max_distance = db_distance
    return max_distance


# In[13]:


def query(db):
    return db.float().mean()


# In[14]:


sensitivity(query, 1000)


# In[15]:


0.1/1000


# In[16]:


#Calculate L1 sensitivity for threshold


# In[17]:


def query(db, threshold=5):
    return (db.sum() > threshold)


# In[18]:


for i in range(10):
    sens_f = sensitivity(query, n_entries=10)
    #print(query)
    print(sens_f)


# In[19]:


#Perform differencing attack on row 10


# In[20]:


db,_ = create_db_and_parallels(100)


# In[21]:


db


# In[22]:


pdb = get_parallel_db(db, remove_index=10)


# In[23]:


db[10]


# In[24]:


sum(db) - sum(pdb)


# In[25]:


sum(db)


# In[26]:


#differencing attack using sum query
sum(db) - sum(pdb)


# In[27]:


#differencing attack using mean query
(sum(db).float() / len(db)) - (sum(pdb).float() / len(pdb))


# In[28]:


#differencing attack using mean query
(sum(db).float() > 49) - (sum(pdb).float() > 49)


# In[29]:


#Implement local differential privacy


# In[30]:


db, pdbs = create_db_and_parallels(100)


# In[31]:


true_result = torch.mean(db.float())
true_result


# db

# In[32]:


first_coin_flip = (torch.rand(len(db)) > 0.5).float()
second_coin_flip = (torch.rand(len(db)) > 0.5).float()
first_coin_flip


# In[33]:


second_coin_flip


# ###Now The first coin flip will determine whether we will use the original value in the database or will use the value that is randonly generated in the second coin flip. We can do this by simply multiplying the first coin flip to the db. because If the first coin flip value is 1 that means the corrsponding answer in the db is true. If it is zero, we don't take the corresponding db value. 
# 
# But sometimes people will lie.We need to go for second coin flip. 

# In[34]:


db.float() * first_coin_flip


# In[35]:


(1-first_coin_flip)


# ###Sometimes we need to choose randomly. in the abobe mention result where there is a 1 that means we need to choose randomly through a second coin flip.

# In[36]:


(1-first_coin_flip) * second_coin_flip


# If we just add them together, we will get a new augmented database that is differentially private

# In[37]:


augmented_databse = (db.float() * first_coin_flip) + ((1-first_coin_flip) * second_coin_flip)


# In[38]:


augmented_databse


# In[39]:


torch.mean(augmented_databse.float())


# We know that half the time the result was honest, and rest half of the time it was random. That in that rest hlf time there is 50/50 chance that result is honest. That gives around 70% honest answer. The taboo behaviour was 70% too. But because of that rest 50% random result, our result will always try to scew towards 50%. Now let's try to deskew the results

# In[40]:


torch.mean(augmented_databse.float()) * 2 - 0.5


# So This is the original result. Let's make a function

# In[41]:


def query(db):
    true_result = torch.mean(db.float())
    first_coin_flip = (torch.rand(len(db)) > 0.5).float()
    second_coin_flip = (torch.rand(len(db)) > 0.5).float()
    augmented_databse = db.float() * first_coin_flip + (1-first_coin_flip) * second_coin_flip
    dp_result = torch.mean(augmented_databse.float()) * 2 - 0.5
    return dp_result, true_result  


# In[42]:


query(db)


# In[43]:


db, pdbs = create_db_and_parallels(10)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# In[44]:


db, pdbs = create_db_and_parallels(100)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# In[45]:


db, pdbs = create_db_and_parallels(1000)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# In[46]:


db, pdbs = create_db_and_parallels(10000)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# If you notice, in this results, bigger the database smaller the effect os noise. When we are adding noise, we are corrupting the dataset. It definitely has an impact on the query. However, the more datapoints we have, the noise effect is tend to average out by the number of data. Local differential privacy is more data hungry. Because we add noise in every data point. To average it out it requires more data. On the other hand global differential privacy is less data hungry, because we add noise only in the output. So the query is less corrupted and more accurate. When the database is not too big, it's a better idea to use global differential privacy.

# The advantage of the local DP is we don't have to tust someone with our data and also we can publish our individual data. 

# In[47]:


def query(db):
    true_result = torch.mean(db.float())
    first_coin_flip = (torch.rand(len(db)) > 0.5).float()
    second_coin_flip = (torch.rand(len(db)) > 0.5).float()
    augmented_databse = db.float() * first_coin_flip + (1-first_coin_flip) * second_coin_flip
    dp_result = torch.mean(augmented_databse.float()) * 2 - 0.5
    return dp_result, true_result  


# We want to get the same result after noise as it was before adding noise. So we need to deskew it correctly. 

# In[48]:


noise=0.2
true_result = torch.mean(db.float())
first_coin_flip = (torch.rand(len(db)) > 0.5).float()
second_coin_flip = (torch.rand(len(db)) > 0.5).float()
augmented_databse = db.float() * first_coin_flip + (1-first_coin_flip) * second_coin_flip
sk_result = augmented_databse.float().mean()


# In[49]:


sk_result


# In[50]:


true_dist_mean = 0.7 #70% of people said yes
noise_dist_mean = 0.5 #50/50 coin flip


# In[51]:


augmented_database_mean = (true_dist_mean + noise_dist_mean) / 2
augmented_database_mean


# In[52]:


noise = 0.5
augmented_database_mean = (true_dist_mean * noise) + noise_dist_mean * (1 - noise)
augmented_database_mean


# So half the time we add noise to the trure distribution. Rets half the time we use noise distribution. So the private result will be something like this:

# In[53]:


private_result = ((sk_result / noise) - 0.5) * noise / (1 - noise)
private_result


# In this private result we are deskewing the result adjusted according to the noise parameter. 

# In[54]:


sk_result


# In[55]:


private_result


# sk_result is basically half way between the private result and the mean of the 50/50 coin flip.

# In[56]:


0.5 - 0.5108


# In[57]:


0.5-0.5216


# .011 is half way from .022. Because it was 50/50 average

# In[58]:


def query(db, noise = 0.2):
    true_result = torch.mean(db.float())
    first_coin_flip = (torch.rand(len(db)) > 0.5).float()
    second_coin_flip = (torch.rand(len(db)) > 0.5).float()
    augmented_databse = db.float() * first_coin_flip + (1-first_coin_flip) * second_coin_flip
    sk_result = augmented_databse.float().mean()
    private_result = ((sk_result / noise) - 0.5) * noise / (1 - noise)
    db_result = torch.mean(augmented_databse.float()) * 2 - 0.5
    return db_result, true_result  


# In[59]:


db, pdbs = create_db_and_parallels(10)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# In[60]:


db, pdbs = create_db_and_parallels(100)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# In[61]:


db, pdbs = create_db_and_parallels(1000)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# In[62]:


db, pdbs = create_db_and_parallels(10000)
private_result, true_result = query(db)
print("With Noise:" + str(private_result))
print("Without Noise:" + str(true_result))


# So bigger the dataset smaller the efefct of noise that means even if you add noise but still you can get the correct results. In society people tend to give statistician or machine learning engineers less and less data to protect privacy. But in practice we can perform better with bigger dataset while keeping the privacy of people. Our intention is not to know about any individual person. For example if we are studying the images of ten thousand tumors to learn the pattern of tumors, we have no interest to learn about any specific person. We are trying to learn about the general trends. So, in differential privacy if we find any data that is very unique we filter out that specific data and work with the data that helps us learn about the trend.  

# In[63]:


#Formal definition of differential privacy


# How much nise should we add after the qquery has been run? 

# In[64]:


#Create a differentially private query


# In[65]:


epsilon = 0.5


# In[66]:


import numpy as np


# In[67]:


db, pdbs = create_db_and_parallels(100)


# In[68]:


def sum_query(db):
    return db.sum()


# In[69]:


def laplacian_mechanism(db, query, sensitivity):
    beta = sensitivity / epsilon
    noise = torch.tensor(np.random.laplace(0, beta, 1))
    return query(db) + noise


# In[70]:


def mean_query(db):
    return torch.mean(db.float())


# In[71]:


torch.tensor(np.random.laplace(0, 5, 1))


# In[72]:


sum_query(db)


# In[73]:


laplacian_mechanism(db, sum_query, 1)


# In[74]:


laplacian_mechanism(db, mean_query, 1/100)


# In[75]:


#Differential privacy for deep learning


# In[76]:


num_teachers = 10 #we're working with 10 partner hospitals
num_examples = 10000 #the size of our dataset
num_labels = 10 #number of labels for our classifier


# In[77]:


preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int) #fake predictions
preds


# In[78]:


an_image = preds[:, 0]
an_image


# In[79]:


label_counts = np.bincount(an_image, minlength=num_labels)


# In[80]:


label_counts


# In[81]:


np.argmax(label_counts)


# In[82]:


#Tis result is not differentially private. So let add some laplacian noise


# In[83]:


epsilon = 0.1
beta = 1 / epsilon
for i in range(len(label_counts)):
    label_counts[i] += np.random.laplace(0, beta, 1)


# In[84]:


label_counts


# In[85]:


np.argmax(label_counts)


# In[86]:


#The noise got us a wrong answer. But this is the price we pay for privacy


# In[87]:


#All this time we worked only on one image. Now let's work on all the data


# In[88]:


new_labels = list()
for an_image in preds:
    label_counts = np.bincount(an_image, minlength=num_labels)
    epsilon = 0.1
    beta = 1 / epsilon
    for i in range(len(label_counts)):
        label_counts[i] += np.random.laplace(0, beta, 1)
    new_label = np.argmax(label_counts)
    new_labels.append(new_label)


# In[89]:


#Pate Analysis


# In[90]:


labels = np.array([9,9,3,6,9,9,9,9,8,2])
counts = np.bincount(labels, minlength=10)
counts


# In[91]:


from syft.frameworks.torch.differential_privacy import pate


# In[92]:


num_teachers = 10 #we're working with 10 partner hospitals
num_examples = 10000 #the size of our dataset
indices = (np.random.rand(num_examples) * num_labels).astype(int)


# In[93]:


data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds = preds, indices = indices, noise_eps = 0.1, delta=1e-5)
print("Data Independent Epsilon: ", data_ind_eps)
print("Data Dependent Epsilon: ", data_dep_eps)


# In[94]:


preds[:, 0:5] *= 0


# In[95]:


data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds = preds, indices = indices, noise_eps = 0.1, delta=1e-5)
print("Data Independent Epsilon: ", data_ind_eps)
print("Data Dependent Epsilon: ", data_dep_eps)


# In[96]:


preds[:, 0:50] *= 0


# In[97]:


data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds = preds, indices = indices, noise_eps = 0.1, delta=1e-5, moments=20)
print("Data Independent Epsilon: ", data_ind_eps)
print("Data Dependent Epsilon: ", data_dep_eps)


# In[ ]:




