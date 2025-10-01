import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. Carga y Preparación de Datos ---

# Se carga el dataset de pruebas
DATA_FILE = "pruebas.csv" 
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, DATA_FILE)
    if not os.path.exists(data_path):
        data_path = DATA_FILE

    df = pd.read_csv(data_path)
    print(f"✅ Datos cargados correctamente desde: {data_path}")
    print(f"Total de registros: {len(df)}")
    print(f"Número de rutinas únicas: {df['Rutina'].nunique()}")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo '{DATA_FILE}'. Asegúrate de que esté en la misma carpeta que el script.")
    exit()

# --- 2. Preprocesamiento y Codificación ---

df_encoded = df.copy()
encoders = {}

categorical_cols = df.select_dtypes(include=['object']).columns

for column in categorical_cols:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    encoders[column] = le

X = df_encoded.drop('Rutina', axis=1)
y = df_encoded['Rutina']

# Aqui se borrarán los registros que solo aparezcan 1 vez, para no dañar el algoritmo
class_counts = y.value_counts()
single_member_classes = class_counts[class_counts < 2].index

if not single_member_classes.empty:
    rutinas_eliminadas = encoders['Rutina'].inverse_transform(single_member_classes)
    print(f"⚠️  Se encontraron clases con 1 solo miembro. Eliminando {len(rutinas_eliminadas)} registros para poder usar 'stratify'.")
    print(f"Rutinas eliminadas: {list(rutinas_eliminadas)}")
    # Filtramos el DataFrame para excluir esas clases
    df_encoded = df_encoded[~df_encoded['Rutina'].isin(single_member_classes)]
    # Regeneramos X e y a partir del DataFrame filtrado
    X = df_encoded.drop('Rutina', axis=1)
    y = df_encoded['Rutina']

# En caso de que nos quedemos sin datos despues de filtrar se para
if df_encoded.empty:
    print("❌ Error: Después de filtrar clases únicas, el dataset quedó vacío. No se puede continuar.")
    exit()

# --- 3. División de Datos y Entrenamiento del Modelo ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=150, random_state=42, oob_score=True, max_features='sqrt')
model.fit(X_train, y_train)
print("✅ Modelo de Random Forest entrenado.")

# --- 4. Evaluación del Modelo ---

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Precisión del modelo en el conjunto de prueba: {accuracy:.2f}")
if hasattr(model, 'oob_score_'):
    print(f"📊 Precisión Out-of-Bag (OOB): {model.oob_score_:.2f}")

# --- 5. Lógica de la API con Flask  ---

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

def predecir_rutina(objetivo, nivel, dias, tiempo, equipo, edad, sexo):
    feature_columns = ['Objetivo', 'Nivel', 'Dias', 'Tiempo', 'Equipo_Disponible', 'Edad', 'Sexo']
    input_data = pd.DataFrame({
        'Objetivo': [objetivo], 'Nivel': [nivel], 'Dias': [int(dias)],
        'Tiempo': [int(tiempo)], 'Equipo_Disponible': [equipo],
        'Edad': [int(edad)], 'Sexo': [sexo]
    })
    input_data = input_data[feature_columns]
    for column in ['Objetivo', 'Nivel', 'Equipo_Disponible', 'Sexo']:
        try:
            le = encoders[column]
            input_data[column] = le.transform(input_data[column])
        except (ValueError, KeyError) as e:
            valores_esperados = list(encoders.get(column, le).classes_)
            raise ValueError(f"Valor no reconocido para '{column}': '{input_data[column].iloc[0]}'. Valores esperados: {valores_esperados}") from e
    prediccion_codificada = model.predict(input_data)
    rutina_recomendada = encoders['Rutina'].inverse_transform(prediccion_codificada)
    return rutina_recomendada[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['objetivo', 'nivel', 'dias', 'tiempo', 'equipo', 'edad', 'sexo']
        if not data or not all(k in data for k in required_fields):
            return jsonify({"error": f"Faltan datos. Se requieren: {', '.join(required_fields)}."}), 400
        recomendacion = predecir_rutina(
            data['objetivo'], data['nivel'], data['dias'], data['tiempo'],
            data['equipo'], data['edad'], data['sexo']
        )
        return jsonify({'rutina_recomendada': recomendacion})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error inesperado: {type(e).__name__} - {e}")
        return jsonify({"error": "Ocurrió un error interno en el servidor."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", "model": "RandomForestClassifier", "dataset": DATA_FILE,
        "total_samples_after_filtering": len(df_encoded),
        "test_set_accuracy": f"{accuracy:.2f}",
        "oob_accuracy": f"{model.oob_score_:.2f}" if hasattr(model, 'oob_score_') else "N/A"
    })

# --- 6. Ejecución del Servidor ---

if __name__ == '__main__':
    if not os.path.exist(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))
