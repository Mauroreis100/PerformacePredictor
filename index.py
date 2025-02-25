import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Carregar os dados
df = pd.read_csv("Extended_Employee_Performance_and_Productivity_Data.csv")

# Selecionar colunas relevantes
features = ["Monthly_Salary","Work_Hours_Per_Week", "Overtime_Hours", "Projects_Handled", "Sick_Days",
            "Remote_Work_Frequency", "Training_Hours", "Employee_Satisfaction_Score",
            "Team_Size", "Promotions", "Years_At_Company"]
target = "Performance_Score"

# Remover valores nulos
df = df.dropna()

# Separar variáveis explicativas (X) e variável alvo (y)
X = df[features]
y = df[target]

# Normalizar os dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Guardar o modelo e o scaler
joblib.dump(model, "productivity_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Modelo treinado e guardado com sucesso!")
