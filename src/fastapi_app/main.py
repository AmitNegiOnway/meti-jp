from fastapi import FastAPI ,Path ,HTTPException ,Query
from typing import Annotated,Literal,Optional
from pydantic import Field,computed_field , BaseModel,field_validator
import json
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler

with open('models/model.pkl', 'rb') as f:
          model=pickle.load(f)


app=FastAPI()

# LungCap","Age","Height","Smoke","Gender","Caesarean"

class Patient(BaseModel):
        Age:Annotated[int,Field(...,gt=0,lt=120,description='age of the patient')]
        Height:Annotated[float,Field(...,description='hieght of the patient')]
        Smoke:Annotated[Literal['yes','no'],Field(...,description='patient do smoke or not?')]
        Gender:Annotated[Literal['male','female'],Field(...,description='gender of the patient')]
        Caesarean:Annotated[Literal['no','yes'],Field(...,description='patient are born Caesarean or not ?')]



@app.post('/predicts')
def predict_lungcap(data:Patient):
       
       input_df=pd.DataFrame([{
               'Age':data.Age,
                'Height':data.Height,
                'Smoke':1 if data.Smoke =='yes' else 0,
                'Gender':1 if data.Gender == 'male' else 0,
                'Caesarean':1 if data.Caesarean =='yes' else 0
        }])
       
       #scaler=StandardScaler()
       #input_scaled=scaler.fit_transform(input_df)
       
       prediction=model.predict(input_df)[0]
       try:
        prediction = model.predict(input_df)[0]
        return JSONResponse(status_code=200, content={"Lung Cap Prediction": float(prediction)})
       except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})





