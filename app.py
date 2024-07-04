from flask import Flask, request, jsonify, render_template, send_file,url_for, session,redirect,sessions as sess
import os
import pyaudio
import wave
import whisper
import io
import tempfile
import openai
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import keras.losses
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

pca = joblib.load('pca_model.pkl')

app = Flask(__name__)

oauth = OAuth(app)
 
# Configure Google OAuth client
google_client_id = os.getenv('GOOGLE_CLIENT_ID')
google_client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
 
oauth.register(
    name='google',
    client_id=google_client_id,
    client_secret=google_client_secret,
    authorize_params=None,
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri=None,
    api_base_url='https://www.googleapis.com/oauth2/v3/',
    client_kwargs={'scope': 'openid profile email', 'prompt': 'consent'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Configuration for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = int(os.getenv('CHANNELS', 1))
RATE = int(os.getenv('RATE', 44100))
CHUNK = int(os.getenv('CHUNK', 1024))
RECORD_SECONDS = int(os.getenv('RECORD_SECONDS', 5))

# Temporary directory to store audio recordings
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load models and scaler
model = whisper.load_model(os.getenv('MODEL_PATH', 'base'))
# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.7)
agent_executor = None
# Load LSTM model with custom_objects to handle custom loss function
try:
    lstm_model = load_model(os.getenv('LSTM_MODEL_PATH', 'lstm_model.h5'), custom_objects={'mse': keras.losses.mean_squared_error})
except ValueError as e:
    print(f"Error loading LSTM model: {e}")

rf_model = joblib.load(os.getenv('RF_MODEL_PATH', 'best_random_forest_model.pkl'))
scaler = joblib.load(os.getenv('SCALER_PATH', 'scaler.pkl'))
kmeans_model = joblib.load('kmeans_model_withpca.pkl')




# Define global variables to hold prediction results
results = []
rmse = 0.0
green_count = 0
red_count = 0
output = io.StringIO()  # Initialize output as StringIO
@app.route('/icreate')
def icreate():
    user_name = session.get('user_name', 'Guest')  # Get user name from session, default to 'Guest' if not available
    return render_template('dashboard.html', user_name=user_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trans', methods=['POST'])
def trans():
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
        
@app.route('/rulpredictions', methods=['GET','POST'])
def rulpredictions():
    return render_template('rulpredictions.html')

@app.route('/langchain', methods=['GET','POST'])
def langchain():
    return render_template('langchain_index.html')

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

        # Apply PCA transformation
        X_pca = pca.transform(X_scaled)

        if model_type == 'LSTM':
            # Reshape the data to fit the LSTM input requirements
            X_scaled_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

            # Predict RUL for uploaded data
            y_pred = lstm_model.predict(X_scaled_lstm)

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

        # Predict conditions using PCA-transformed data
        conditions = kmeans_model.predict(X_pca)  # Use PCA-transformed data
        condition_labels = ['Healthy' if cond == 0 else 'Not Healthy' for cond in conditions]  # Adjust the condition labels based on your KMeans clustering

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
                'urgency': urgency,
                'condition': condition_labels[i]  # Add condition to results
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
                    result['urgency'] = "NIL"

        # Write the sorted results to output CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(output, index=False)
        output.seek(0)  # Move to the start of the StringIO object

        return render_template('results.html', rmse=rmse, results=results, csv_data=output.getvalue(),
                               green_count=green_count, red_count=red_count)

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/download_csv')
def download_csv():
    global output

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
    
@app.route('/google/')
def google():
    redirect_uri = url_for('google_auth', _external=True)
    session['nonce'] = generate_token()
    return oauth.google.authorize_redirect(redirect_uri, nonce=session['nonce'])

 
@app.route('/google/auth/')
def google_auth():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token, nonce=session['nonce'])
    session['user_name'] = user_info.get('name', 'Guest')  # Store user name in session
    return redirect('/icreate')
 
 
@app.route('/query')
def query_page():
    return render_template('query.html')

@app.route('/query', methods=['POST'])
def query():
    global agent_executor

    user_input = request.form['query']
    if agent_executor:
        result = str(agent_executor.invoke(user_input))
        re = result
        #Creating a mp3 file with the output
        client = openai.OpenAI()

        speech_file_path = Path().parent / "hi.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=re
        )
        response.stream_to_file(speech_file_path)
        return jsonify(result=result)
    else:
        return jsonify(result="Please upload a CSV file first.")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        global agent_executor
        agent_executor = create_csv_agent(llm, filepath, verbose=True, allow_dangerous_code=True)
        return redirect(url_for('query_page'))
    return redirect(request.url)

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

            # Process text using agent_executor
            output = str(agent_executor.invoke(text_en))
            re = output
            #Creating a mp3 file with the output
            client = openai.OpenAI()

            speech_file_path = Path().parent / "hi.mp3"
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=re
                )
            response.stream_to_file(speech_file_path)
            
            # Return JSON response with both transcription_en and output
            return jsonify(transcription_en=text_en, output=output)

        except Exception as e:
            print(f"Error during transcription: {e}")
            return str(e), 500

@app.route('/get_audio')
def get_audio():
    # Replace 'path_to_your_audio_file.mp3' with the actual path to your MP3 file
    return send_file('hi.mp3', mimetype='audio/mpeg')


if __name__ == "__main__":
    app.secret_key = os.getenv('SECRET_KEY')
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app.run(debug=True)
