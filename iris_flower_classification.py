import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\manik\\OneDrive\\Documents\\codsoft\\iris\\IRIS.csv")
df.head()

df.isnull().sum()  #checking for null values

the columns we have are 

df.columns

dividing the input variable and the target variable 

x = df.iloc[:,:4]
y = df.iloc[:,4]

x

y

spliting our data into training and testing data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

now i am creating a logistic regression model to classify the flower species

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

now we are checking accuracy score of a model and the confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

score = accuracy_score(y_test,y_pred)*100
score

cm = confusion_matrix(y_test,y_pred)
cm

CONCLUSION : 

- in above i have used a iris flower dataset which has been downloaded from kaggle 
- i have used logistic regression to built a model which has performed vey well
- the accuracy of the model is 100 accurate 
- finally the flowers are classified into their species
