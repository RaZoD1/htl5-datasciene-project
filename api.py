from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import RandomForestModel
import pandas as pd

app = FastAPI()

class NetworkData(BaseModel):
    duration: int
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float


model = RandomForestModel("https://docs.google.com/spreadsheets/d/e/2PACX-1vQkqK3rzUUOf-RIkiSU5RszMzHVwYgPTJUek6qjDrW6_F3MyJ-eETUa5UgiRzNdt6PhFtcKI6gioaj6/pub?gid=1746802197&single=true&output=csv")
model.run()


@app.post("/predict")
async def predict(request: NetworkData):
    # Convert the incoming request data to a dictionary with only the fields from the model
    data_dict = request.model_dump()
    
    # Convert the dictionary to a pandas DataFrame (one row)
    input_data = pd.DataFrame([data_dict])

    # Perform prediction
    prediction = model.predict(input_data)[0]

    print(prediction)

    return {"prediction": prediction}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
