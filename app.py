from flask import Flask, request, jsonify, render_template, send_file
import os
import pyaudio
import wave
import whisper
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
model = whisper.load_model("base")
lstm_model = load_model('lstm_model.h5')
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')   

# Configuration for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust as needed

# Temporary directory to store audio recordings
UPLOAD_FOLDER = '/tmp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define global variables to hold prediction results
results = []
rmse = 0.0
green_count = 0
red_count = 0
output = io.StringIO()  # Initialize output as StringIO

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return "No file part", 400

    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"Saved file to {filepath}")

        try:
            # Transcribe audio forcing English language
            result = model.transcribe(filepath, language='en')
            text_en = result['text']
            print(f"Transcription (English): {text_en}")

            os.remove(filepath)
            print(f"Removed file {filepath}")

            return jsonify({"transcription_en": text_en})

        except Exception as e:
            print(f"Error during transcription: {e}")
            return str(e), 500

@app.route('/predict', methods=['POST'])
def predict():
    global results, rmse, green_count, red_count, output
    
    # Reset output as StringIO before processing
    output = io.StringIO()

    # Get the uploaded file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    uploaded_file = request.files['file']
    model_type = request.form['model']

    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Load the CSV data
        data = pd.read_csv(uploaded_file)

        # Drop columns that are not used for prediction
        data = data.drop(['unit_ID', 'cycles'], axis=1)

        # Standardize the features
        X = data.drop('RUL', axis=1)
        X_scaled = scaler.transform(X)

        if model_type == 'LSTM':
            # Reshape the data to fit the LSTM input requirements
            X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

            # Predict RUL for uploaded data
            y_pred = lstm_model.predict(X_scaled)

        elif model_type == 'RF':
            # Predict RUL using Random Forest model
            y_pred = rf_model.predict(X_scaled)
            y_pred = y_pred.reshape(-1, 1)

        elif model_type == 'Combined':
            # Reshape the data to fit the LSTM input requirements
            X_scaled_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

            # Predict RUL using both models
            y_pred_lstm = lstm_model.predict(X_scaled_lstm)
            y_pred_rf = rf_model.predict(X_scaled).reshape(-1, 1)

            # Combine the predictions (e.g., average)
            y_pred = (y_pred_lstm + y_pred_rf) / 2

        else:
            return jsonify({'error': 'Invalid model type selected.'}), 400

        # Calculate y_true here after dropping and processing data
        y_true = data['RUL'].values

        # Prepare response
        results = []
        green_count = 0
        red_count = 0

        # Collect all green actual RULs for ranking
        green_actual_ruls = []

        for i in range(len(y_pred)):
            predicted_RUL = float(y_pred[i])
            actual_RUL = float(y_true[i])
            urgency = 0  # Initialize urgency as 0

            if actual_RUL >= predicted_RUL:
                green_count += 1
                green_actual_ruls.append(actual_RUL)
            else:
                red_count += 1

            results.append({
                'index': i,
                'predicted_RUL': predicted_RUL,
                'actual_RUL': actual_RUL,
                'urgency': urgency
            })

        # Calculate RMSE (example, adjust as needed)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Rank the green boxes based on actual RUL
        if green_count > 0:
            # Sort green_actual_ruls to determine ranking
            ranked_green_ruls = sorted(green_actual_ruls)

            for result in results:
                if result['actual_RUL'] >= result['predicted_RUL']:
                    # Find the rank of this green box
                    rank = ranked_green_ruls.index(result['actual_RUL']) + 1
                    result['urgency'] = rank
                else:
                    result['urgency'] = "NIL"  # Red boxes get "NIL" urgency

        # Save results to a CSV file
        df_results = pd.DataFrame(results)
        df_results.to_csv(output, index=False)
        output.seek(0)  # Move cursor to start of StringIO buffer

        return render_template('results.html', rmse=rmse, results=results, csv_data=output.getvalue(),
                               green_count=green_count, red_count=red_count)

    except Exception as e:
        return jsonify({'error': f'Error processing prediction: {str(e)}'}), 500

@app.route('/download_results')
def download_results():
    global output
    
    # Ensure output is initialized as StringIO if needed
    if not isinstance(output, io.StringIO):
        output = io.StringIO()

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predicted_results.csv'
    )

@app.route('/sort')
def sort_results():
    global results, rmse, green_count, red_count, output
    
    column = request.args.get('col', 'index')  # Default column to sort by 'index'

    # Ensure output is initialized as StringIO if needed
    if not isinstance(output, io.StringIO):
        output = io.StringIO()

    # Sort the results based on the selected column
    if column == 'index':
        results_sorted = sorted(results, key=lambda x: int(x[column]))  # Convert to int for sorting
    elif column in ['predicted_RUL', 'actual_RUL']:
        results_sorted = sorted(results, key=lambda x: float(x[column]))  # Convert to float for sorting
    elif column == 'urgency':
        results_sorted = sorted(results, key=lambda x: float('inf') if x[column] == 'NIL' else float(x[column]))
    else:
        results_sorted = sorted(results, key=lambda x: x[column])  # Default to string sorting

    return render_template('results.html', rmse=rmse, results=results_sorted, csv_data=output.getvalue(),
                           green_count=green_count, red_count=red_count)


if __name__ == "__main__":
    app.run(debug=True)
