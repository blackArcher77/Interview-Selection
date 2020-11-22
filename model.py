import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("selection.csv")
#print(data.head())

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

model=LogisticRegression()
model.fit(x,y)

pickle.dump(model,open('model.pkl','wb'))


#xt=np.array([8.9,20]).reshape(1,-1)
#print(model.predict(xt))
