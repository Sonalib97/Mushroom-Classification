#!/usr/bin/env python
# coding: utf-8

# # Mushroom Classification

# In[ ]:


#Load the  libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn


# In[2]:


#Load the dataset
df=pd.read_csv("C:\\Users\\ASUS\\Downloads\\mushrooms.csv")
df


# In[3]:


#show the dimension of the dataset
df.shape

#So there are 8124 rows and 23 columns in the data.
# In[4]:


#show the top 5 records
df.head()


# In[5]:


#show the bottom 5 records
df.tail()


# In[6]:


#show 5 records randomly
df.sample(5)


# # EDA
EDA was initally performed on the dataset before one-hot encoding was applied to understand categorical distributions of features.Feature value key:
    
1.cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2.cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3.cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
4.bruises: bruises=t,no=f
5.odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
6.gill-attachment: attached=a,descending=d,free=f,notched=n
7.gill-spacing: close=c,crowded=w,distant=d
8.gill-size: broad=b,narrow=n
9.gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
10.stalk-shape: enlarging=e,tapering=t
11.stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
12.stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13.stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14.stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
15.stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
16.veil-type: partial=p,universal=u
17.veil-color: brown=n,orange=o,white=w,yellow=y
18.ring-number: none=n,one=o,two=t
19.ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
20.spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
21.population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
22.habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
# In[7]:


#Datatypes present in our data
df.dtypes

So all our data is categorical.
# In[8]:


#Summary of the data
df.info()


# In[9]:


#Overall Statistics of the Dataset.
df.describe()


# In[10]:


#Let's check the classes present in df.
df['class'].unique()

So we have 2 classes here. 'p'-poisonous & 'e'-edible.
# In[11]:


#Check the counts of classes
df['class'].value_counts()


# In[12]:


sns.countplot(df['class'])

So our data is almost balanced.
# In[13]:


#Check for missing values.
df.isnull().sum()

Hence our Dataset has no missing value.
# # Data Manipulation
The data is categorical so we’ll use LabelEncoder to convert it to ordinal. LabelEncoder converts each value in a column to a number.This approach requires the category column to be of ‘category’ datatype. By default, a non-numerical column is of ‘object’ datatype. From the df.describe() method, we saw that our columns are of ‘object’ datatype. So we will have to change the type to ‘category’ before using this approach.
# In[14]:


df=df.astype('category')


# In[15]:


df.dtypes


# In[16]:


#Let's convert categorical values into Numerical values.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in df.columns:
    df[column]=le.fit_transform(df[column])


# In[17]:


df.head()


# # Define Features (X) and Targets(Y)

# In[18]:


X=df.drop(columns='class')
X


# In[19]:


Y=df['class']
Y


# # Applying PCA

# In[21]:


from sklearn.decomposition import PCA
pca1 = PCA(n_components=7)
pca_fit = pca1.fit_transform(X)


# In[22]:


pca1.explained_variance_ratio_


# # Split the Dataset into 2 parts for Training & Testing.

# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=11)


# In[24]:


X_train


# In[25]:


Y_train


# In[26]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# # Import the models

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# # Model Training

# In[28]:


L= LogisticRegression()
L.fit(X_train,Y_train)

knn= KNeighborsClassifier()
knn.fit(X_train,Y_train)

svc= SVC()
svc.fit(X_train,Y_train)

dt= DecisionTreeClassifier()
dt.fit(X_train,Y_train)

rm= RandomForestClassifier()
rm.fit(X_train,Y_train)

gb= GradientBoostingClassifier()
gb.fit(X_train,Y_train)


# # Prediction on Test Data

# In[29]:


y_pred_L= L.predict(X_test)
y_pred_knn= knn.predict(X_test)
y_pred_svc= svc.predict(X_test)
y_pred_dt= dt.predict(X_test)
y_pred_rm= rm.predict(X_test)
y_pred_gb= gb.predict(X_test)


# # Evaluate the Algorithm

# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


print("ACC L:",accuracy_score(Y_test,y_pred_L))
print("ACC KNN:",accuracy_score(Y_test,y_pred_knn))
print("ACC SVC:",accuracy_score(Y_test,y_pred_svc))
print("ACC DT:",accuracy_score(Y_test,y_pred_dt))
print("ACC RM:",accuracy_score(Y_test,y_pred_rm))
print("ACC GBC:",accuracy_score(Y_test,y_pred_gb))


# In[32]:


final_data = pd.DataFrame({'Models':['L','KNN','SVC','DT','RM','GBC'],
             'ACC':[accuracy_score(Y_test,y_pred_L)*100,
                    accuracy_score(Y_test,y_pred_knn)*100,
                    accuracy_score(Y_test,y_pred_svc)*100,
                    accuracy_score(Y_test,y_pred_dt)*100,
                    accuracy_score(Y_test,y_pred_rm)*100,
                    accuracy_score(Y_test,y_pred_gb)*100]})


# In[33]:


final_data


# In[34]:


#Display the Accuracy & model in a bar plot.
import matplotlib.pyplot as plt
plt.bar(final_data['Models'],final_data['ACC'],color=['violet','green','orange','blue','red','pink'])


# # Save the Model

# In[36]:


rf_model= RandomForestClassifier()
rf_model.fit(pca_fit,Y)


# In[37]:


import joblib


# In[41]:


joblib.dump(rf_model,"mushrooms")


# In[42]:


model = joblib.load('mushrooms')


# In[44]:


p= model.predict(pca1.transform([[5,2,4,1,6,1,0,1,4,0,3,2,2,7,7,0,2,1,4,2,3,5]]))


# In[46]:


if p[0]==1:
    print('Poissonous')
else:
    print('Edible')


# # GUI

# In[47]:


from tkinter import *
import joblib


# In[49]:


def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get()) 
    p10=int(e10.get())
    p11=int(e11.get())
    
    p12=int(e12.get())
    p13=int(e13.get())
    p14=int(e14.get())
    p15=int(e15.get())
    p16=int(e16.get())
    p17=int(e17.get())
    p18=int(e18.get())
    p19=int(e19.get())
    p20=int(e20.get())
    p21=int(e21.get())
    p22=int(e22.get())
    
    model = joblib.load('Mushroom_prediction')
    result=model.predict(pca1.transform([[p1,p2,p3,p4,p5,p6,
                           p7,p8,p9,p10,p11,p12,p13,p14,p15,
                            p16,p17,p18,p19,p20,p21,p22]]))
    if result[0] == 0:
        Label(master, text="Edible").grid(row=31)
    else:
        Label(master, text="Poisonous").grid(row=31)
    
    
master = Tk()
master.title("Mushroom Classification Using Machine Learning")


label = Label(master, text = "Mushroom Classification Using Machine Learning"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master,text="cap-shape :(cap-shape: bell=0,conical=1,convex=5,flat=2, knobbed=3,sunken=4)").grid(row=1)
Label(master, text="cap-surface:(fibrous=0,grooves=1,scaly=3,smooth=2)").grid(row=2)
Label(master, text="cap-color:(brown=4,buff=0,cinnamon=1,gray=3,green=r, pink=5,purple=6,red=2,white=7,yellow=8)").grid(row=3)
Label(master, text="bruises:(bruises=1,no=0)").grid(row=4)
Label(master, text="odor:(almond=0,anise=3,creosote=1,fishy=8,foul=2,musty=4,none=5,pungent=6,spicy=7 )").grid(row=5)
Label(master, text="gill-attachment:(attached=0,descending=1,free=2,notched=3)").grid(row=6)
Label(master, text="gill-spacing:(close=0,crowded=2,distant=1 )").grid(row=7)
Label(master, text="gill-size:(road=0,narrow=1)").grid(row=8)
Label(master, text="gill-color:(black=4,brown=5,buff=0,chocolate=3,gray=2,green=8,orange=6,pink=7,purple=9,red=1,white=10,yellow=11)").grid(row=9)
Label(master, text="stalk-shape:(enlarging=0,tapering=1)").grid(row=10)
Label(master,text="stalk-root:( bulbous=0,club=1,cup=5,equal=2,rhizomorphs=4, rooted=3,missing=6)").grid(row=11)
Label(master,text="stalk-surface-above-ring:(fibrous=0,scaly=3,silky=1,smooth=2)").grid(row=12)
Label(master,text="stalk-surface-below-ring:(fibrous=0,scaly=3,silky=1,smooth=2 )").grid(row=13)
Label(master,text="stalk-color-above-ring:(brown=4,buff=0,cinnamon=1,gray=3, orange=5,pink=6,red=2,white=7,yellow=8)").grid(row=14)
Label(master,text="stalk-color-below-ring:(brown=4,buff=0,cinnamon=1,gray=3, orange=5,pink=6,red=2,white=7,yellow=8)").grid(row=15)
Label(master,text="veil-type:(partial=0,universal=1)").grid(row=16)
Label(master,text="veil-color:(brown=0,orange=1,white=2,yellow=3)").grid(row=17)
Label(master,text="ring-number:(none=0,one=1,two=2)").grid(row=18)
Label(master,text="ring-type:(cobwebby=0,evanescent=1,flaring=2,large=3,none=4,pendant=5,sheathing=6,zone=7)").grid(row=19)
Label(master,text="spore-print-color:(black=2,brown=3,buff=0,chocolate=1, green=5,orange=4,purple=6,white=7,yellow=8 )").grid(row=20)

Label(master,text="population:(abundant=0,clustered=1,numerous=2,scattered=3, # several=4,solitary=5)").grid(row=21)
Label(master,text="habitat:(grasses=1,leaves=2,meadows=3,paths=4,urban=5,# waste=6,woods=0)").grid(row=22)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)

e12 = Entry(master)
e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18 = Entry(master)
e19 = Entry(master)
e20 = Entry(master)
e21 = Entry(master)
e22 = Entry(master)



e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10,column=1)
e11.grid(row=11,column=1)

e12.grid(row=12,column=1)
e13.grid(row=13,column=1)
e14.grid(row=14,column=1)
e15.grid(row=15,column=1)
e16.grid(row=16,column=1)
e17.grid(row=17,column=1)
e18.grid(row=18,column=1)
e19.grid(row=19,column=1)
e20.grid(row=20,column=1)
e21.grid(row=21,column=1)
e22.grid(row=22,column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# # Thank You
