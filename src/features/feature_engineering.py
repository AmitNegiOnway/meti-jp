import pandas as pd 
import os 
from sklearn.preprocessing import StandardScaler


train_data=pd.read_csv('data/processed/train_data.csv')
test_data=pd.read_csv('data/processed/test_data.csv')


train_data['Smoke']=train_data['Smoke'].map({'no':0,'yes':1})
train_data['Caesarean']=train_data['Caesarean'].map({'no':0,'yes':1})
train_data['Gender']=train_data['Gender'].map({'male':1,'female':0})



test_data['Smoke']=test_data['Smoke'].map({'no':0,'yes':1})
test_data['Caesarean']=test_data['Caesarean'].map({'no':0,'yes':1})
test_data['Gender']=test_data['Gender'].map({'male':1,'female':0})

X_train=train_data.iloc[:,1:]
X_test=test_data.iloc[:,1:]


y_train=train_data.iloc[:,0]
y_test=test_data.iloc[:,0]

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

data_path=os.path.join('data','interim')
os.makedirs(data_path,exist_ok=True)
os.makedirs("data/interim",exist_ok=True)

pd.DataFrame(X_train_scaled,columns=X_train.columns).to_csv(os.path.join(data_path,'X_train_scaled.csv'),index=False)
pd.DataFrame(X_test_scaled,columns=X_test.columns).to_csv(os.path.join(data_path,'X_test_scaled.csv'),index=False)

y_train.to_csv(os.path.join(data_path,'y_train.csv'),index=False)
y_test.to_csv(os.path.join(data_path,'y_test.csv'),index=False)




