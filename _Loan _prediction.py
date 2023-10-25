#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

loan=pd.read_excel("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\MAJOR-PROJECT.xlsx")
df=pd.DataFrame(loan)


# In[2]:


df.isnull().sum()


# In[3]:


df.head()


# In[4]:


import seaborn as sns

sns.countplot(x="Loan_status",hue="Loan_status",data=loan)


# In[5]:


sns.countplot(x="Gender",hue="Gender",data=loan)


# In[6]:


df.head()


# In[7]:


df=df[["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_history","Loan_status"]]
df.head()


# In[8]:


features = ["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_history"]

x=df[features]
y=df["Loan_status"]


# In[9]:


print(x.shape)
print(y.shape)


# In[10]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[11]:


print(x_train.shape)
print(y_train.shape)


# In[12]:


print(x_test.shape)
print(y_test.shape)


# In[13]:


from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()
classifier.fit(x_train,y_train)

classifier.fit(x_test,y_test)


# In[14]:


df.head()


# In[15]:


y_pred = classifier.predict(x_test)


print("y_pred:",y_pred)


# In[16]:


from sklearn.metrics import  accuracy_score

accurancy_test=accuracy_score(y_test,y_pred)*100
print("accurancy_test:",accurancy_test)


# In[17]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print("cm")
print(cm)


# In[ ]:





# In[ ]:





# In[6]:


import pandas as pd

# Load the Excel file into a DataFrame
loan=pd.read_excel("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\MAJOR-PROJECT.xlsx")
df=pd.DataFrame(loan)

# Check for missing values
df.isnull().sum()


# Select relevant columns for analysis
df=df[["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_history","Loan_status"]]


# Define the features (independent variables) and the target variable (dependent variable)
features = ["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_history"]

x=df[features]
y=df["Loan_status"]


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# Initialize the   Decision tree classifier 
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier()

# Train the model using the training data
classifier.fit(x_train,y_train)

 
# Predict the target variable on the test data
y_pred = classifier.predict(x_test)
print("y_pred:",y_pred)



# Calculate the accuracy of the model's predictions on the test dat
from sklearn.metrics import  accuracy_score
accurancy_test=accuracy_score(y_test,y_pred)*100
print("accurancy_test:",accurancy_test)
 

    
# Calculate the confusion matrix to evaluate the model's performance
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print("cm")
print(cm) 


# In[210]:


prediction_data=pd.DataFrame({"Dependents":[2],
                            "ApplicantIncome":[9600000],
                            "CoapplicantIncome":[9500000],
                            "LoanAmount":[29900000],
                            "Loan_Amount_Term":[12],
                            "Credit_history":[778]
                            })


# In[6]:


import pandas as pd

# Load the Excel file into a DataFrame
loan=pd.read_excel("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\MAJOR-PROJECT.xlsx")
df=pd.DataFrame(loan)

# Check for missing values
df.isnull().sum()


# Select relevant columns for analysis
df=df[["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_history","Loan_status"]]


# Define the features (independent variables) and the target variable (dependent variable)
features = ["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_history"]

x=df[features]
y=df["Loan_status"]


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# Initialize the logistic regression classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=5)

# Train the model using the training data
classifier.fit(x_train,y_train)

 
# Predict the target variable on the test data
y_pred = classifier.predict(x_test)
print("y_pred:",y_pred)



# Calculate the accuracy of the model's predictions on the test dat
from sklearn.metrics import  accuracy_score
accurancy_test=accuracy_score(y_test,y_pred)
print("accurancy_test:",accurancy_test)
 

    
# Calculate the confusion matrix to evaluate the model's performance
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print("cm")
print(cm)


# In[ ]:





# In[ ]:




