import joblib
import numpy as np

# Carregar o modelo treinado e o scaler
model = joblib.load("productivity_model.pkl")
scaler = joblib.load("scaler.pkl")

# Novo colaborador (substituir pelos dados reais)
new_employee = np.array([[3000,40, 5, 15, 2, 50, 20, 3.5, 10, 1, 5]])

# Normalizar os dados
new_employee_scaled = scaler.transform(new_employee)

# Fazer previsão
predicted_score = model.predict(new_employee_scaled)
print(f"Produtividade prevista: {predicted_score[0]:.2f}")
