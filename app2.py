from fastapi import FastAPI
import joblib
import numpy as np

# Carregar o modelo treinado e o scaler
model = joblib.load("productivity_model.pkl")
scaler = joblib.load("scaler.pkl")

# Criar a aplicação FastAPI
app = FastAPI()

@app.get("/predict")
def predict(
    Monthly_Salary:float, Work_Hours_Per_Week: float, Overtime_Hours: float, Projects_Handled: float,
    Sick_Days: float, Remote_Work_Frequency: float, Training_Hours: float,
    Employee_Satisfaction_Score: float, Team_Size: float, Promotions: float,
    Years_At_Company: float
):
    try:
        # Criar um array com os dados fornecidos na query string
        new_employee = np.array([[Monthly_Salary,Work_Hours_Per_Week, Overtime_Hours, Projects_Handled, 
                                  Sick_Days, Remote_Work_Frequency, Training_Hours, 
                                  Employee_Satisfaction_Score, Team_Size, Promotions, 
                                  Years_At_Company]])
        
        # Normalizar os dados. IT -> 4
        new_employee_scaled = scaler.transform(new_employee)
        
        # Fazer previsão
        predicted_score = model.predict(new_employee_scaled)

        return {"Produtividade_Prevista": round(predicted_score[0], 2)}

    except Exception as e:
        return {"error": str(e)}


