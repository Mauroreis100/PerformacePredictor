from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import pandas as pd

app = Flask(__name__)

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"

# Load Model & Scaler
model = joblib.load("productivity_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['GET', 'POST'])  
def predict():
    try:
        if request.method == 'GET':  
            monthly_salary = float(request.args.get("Monthly_Salary", 0))
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

            # Create feature array
            employee_data = np.array([[monthly_salary, work_hours, overtime, projects, sick_days, 
                                       remote_freq, training_hours, satisfaction, team_size, 
                                       promotions, years_at_company]])
            employee_data_scaled = scaler.transform(employee_data)
            predicted_score = model.predict(employee_data_scaled)

            return jsonify({"predicted_productivity": round(predicted_score[0], 2)})

        elif request.method == 'POST':  
            data = request.json
            df = pd.DataFrame([data])

            # Ensure features are in the correct order
            feature_order = ["Monthly_Salary", "Work_Hours_Per_Week", "Overtime_Hours", 
                             "Projects_Handled", "Sick_Days", "Remote_Work_Frequency", 
                             "Training_Hours", "Employee_Satisfaction_Score", "Team_Size", 
                             "Promotions", "Years_At_Company"]
            
            df = df[feature_order]  
            df_scaled = scaler.transform(df)

            # Predict Performance Score
            prediction = model.predict(df_scaled)[0]

            # Send prediction result back to Slack
            slack_message = {
                "text": f"📊 *Predicted Performance Score:* `{prediction}`"
            }
            slack_response = requests.post(SLACK_WEBHOOK_URL, json=slack_message)

            if slack_response.status_code != 200:
                print("⚠️ Slack Error:", slack_response.text)

            return jsonify({"predicted_performance": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  
