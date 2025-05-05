from fastapi import FastAPI,UploadFile,File
import pandas as pd
from model import model, encoder
from io import StringIO

app=FastAPI()

@app.get("/")
def home():
    return {"message":"Drug classification API is running"}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    if not file.filename.endswith(".csv"):
        return {"message":"File uploaded is not a csv file"}
    
    content=await file.read()
    data_str=content.decode("utf-8")
    new_data=pd.read_csv(StringIO(data_str))

    catagorical_cols=["Sex","BP","Cholesterol"]
    new_data_encoder_array=encoder.transform(new_data[catagorical_cols])
    new_data_encoded=pd.DataFrame(new_data_encoder_array,columns=catagorical_cols)
    featured_cols=["Age","Sex","BP","Cholesterol","Na_to_K"]
    numeric_cols=["Age","Na_to_K"]
    new_data_numeric=new_data[numeric_cols]
    X_test_final=pd.concat([new_data_encoded,new_data_numeric],axis=1)
    X_test_final=X_test_final[featured_cols]

    prediction=model.predict(X_test_final)
    new_data["Drug"]=prediction
    return new_data[["Age","Sex","BP","Cholesterol","Na_to_K","Drug"]].to_dict(orient="records")

