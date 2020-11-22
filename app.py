from flask import Flask,render_template,request
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))

app = Flask("__init__")

@app.route('/',methods=['GET','POST'])
def home():
   if(request.method=="POST"):
      a=float(request.form["CGPA"])
      b=int(request.form["AGE"])
      temp=np.array([a,b]).reshape(1,-1)
      result=model.predict(temp)
      value="X"
      if(result[0]==0):
         value="Not Selected"
      else:
         value="Selected"
      return render_template("index.html",result=value)
   return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)
