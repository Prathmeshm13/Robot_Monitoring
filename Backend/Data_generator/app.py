from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, session
from flask_cors import CORS
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import time
import joblib
import json

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Change to your MySQL username
app.config['MYSQL_PASSWORD'] = 'PrathmeshM@13'  # Change to your MySQL password
app.config['MYSQL_DB'] = 'failure_monitoring'
mysql = MySQL(app)

# Load ML models
regressor = joblib.load("C:/Users/1041210/OneDrive - Blue Yonder/Desktop/FINAL_PYTHON_PROJECT/Backend/Data_generator/models/xgb_regressor.pkl")
classifier = joblib.load("C:/Users/1041210/OneDrive - Blue Yonder/Desktop/FINAL_PYTHON_PROJECT/Backend/Data_generator/models/trained_classifier (1).pkl")

csv_file_path = "C:/Users/1041210/OneDrive - Blue Yonder/Desktop/FINAL_PYTHON_PROJECT/Backend/Data_generator/ur5_motor1_failure_pattern.csv"

data_buffer = []

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({"message": "Email already registered"}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
    mysql.connection.commit()
    cursor.close()

    return jsonify({"message": "Signup successful"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    print(email)
    print(password)
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    print(user)
    if user and check_password_hash(user[2], password):
        session['loggedin'] = True
        session['id'] = user[0]
        session['username'] = user[1]
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"message": "Invalid email or password"}), 401

@app.route('/')
def index():
    if 'loggedin' in session:
        return render_template('index.html', username=session['username'])
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('index'))  # Redirecting to index or login page after logout


@app.route('/stream-data', methods=['GET'])
def stream_data():
    def generate():
        global data_buffer
        for chunk in pd.read_csv(csv_file_path, chunksize=1):
            for _, row in chunk.iterrows():
                try:
                    joint_1_torque = float(row["Joint_1_Torque"])
                    load_variation = float(row["Load_Variation"])
                    joint_1_temp = float(row["Joint_1_Temperature"])
                    ambient_temp = float(row["Ambient_Temperature"])
                    joint_1_vibration = float(row["Joint_1_Vibration"])
                    joint_1_velocity = float(row["Joint_1_Velocity"])
                    voltage_fluctuation = float(row["Voltage_Fluctuation"])

                    σ_eff = joint_1_torque / (load_variation + 1)
                    CDI = σ_eff
                    TDF = np.exp((joint_1_temp - ambient_temp) / 20)
                    VFI = joint_1_vibration * joint_1_velocity
                    Voltage_Impact = voltage_fluctuation / 250

                    data_df = pd.DataFrame([[σ_eff, CDI, TDF, VFI, Voltage_Impact]],
                                           columns=["σ_eff", "CDI", "TDF", "VFI", "Voltage_Impact"])
                    failure_risk_score = regressor.predict(data_df)[0]
                    failure_label = classifier.predict(data_df)[0]

                    timestamp = time.strftime('%H:%M:%S')

                    data_point = {
                        "timestamp": timestamp,
                        "failure_risk_score": round(float(failure_risk_score), 2),
                        "joint_1_torque": round(float(joint_1_torque), 2),
                        "load_variation": round(float(load_variation), 2),
                        "joint_1_temp": round(float(joint_1_temp), 2),
                        "ambient_temp": round(float(ambient_temp), 2),
                        "joint_1_vibration": round(float(joint_1_vibration), 2),
                        "joint_1_velocity": round(float(joint_1_velocity), 2),
                        "voltage_fluctuation": round(float(voltage_fluctuation), 2),
                        "failure_label": int(failure_label)
                    }

                    data_buffer.append(data_point)
                    if len(data_buffer) > 20:
                        data_buffer.pop(0)

                    yield f"data: {json.dumps(data_buffer)}\n\n"
                    time.sleep(1)
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/plot-data', methods=['GET'])
def plot_data():
    return jsonify(data_buffer)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)