
from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import pandas as pd

app = Flask(__name__)

# Slack Webhook URL
SLACK_WEBHOOK_URL = https://hooks.slack.com/services/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

# Carregar o modelo e o scaler
model = joblib.load("productivity_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Obter parâmetros da URL
        monthly_salary=work_hours = float(request.args.get("Monthly_Salary", 0))
        work_hours = float(request.args.get("Work_Hours_Per_Week", 0))
        overtime = float(request.args.get("Overtime_Hours", 0))
        projects = float(request.args.get("Projects_Handled", 0))
        sick_days = float(request.args.get("Sick_Days", 0))
        remote_freq = float(request.args.get("Remote_Work_Frequency", 0))
        training_hours = float(request.args.get("Training_Hours", 0))
        satisfaction = float(request.args.get("Employee_Satisfaction_Score", 0))
        team_size = float(request.args.get("Team_Size", 0))
        promotions = float(request.args.get("Promotions", 0))
        years_at_company = float(request.args.get("Years_At_Company", 0))

        # Criar array com os valores
        employee_data = np.array([[monthly_salary,work_hours, overtime, projects, sick_days, remote_freq,
                                   training_hours, satisfaction, team_size, promotions, years_at_company]])

        # Normalizar os dados
        employee_data_scaled = scaler.transform(employee_data)

        # Fazer previsão
        predicted_score = model.predict(employee_data_scaled)

        # Retornar a previsão em JSON
        return jsonify({"predicted_productivity": round(predicted_score[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        df_scaled = scaler.transform(df)

        # Predict Performance Score
        prediction = model.predict(df_scaled)[0]

        # Send prediction result back to Slack
        slack_message = {
            "text": f"📊 *Predicted Performance Score:* `{prediction}`"
        }
        requests.post(SLACK_WEBHOOK_URL, json=slack_message)

        return jsonify({"predicted_performance": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Executa na porta 5000
