import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import mlflow
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

# Initialize Flask app
app = Flask(__name__)


# Get the best model
def load_best_model():
    """Load the best model from MLflow registry"""
    model_name = "best_classification_model"
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_feature_importances():
    """Get feature importances from best run"""
    try:
        with open("best_run_id.txt", "r") as f:
            run_id = f.read().strip()

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)

        feature_importance_files = [
            art.path for art in artifacts
            if art.path.startswith("feature_importances_")
        ]

        if not feature_importance_files:
            return None

        # Download the artifact
        temp_path = client.download_artifacts(run_id, feature_importance_files[0], ".")
        feature_importances = pd.read_csv(temp_path)
        return feature_importances
    except Exception as e:
        print(f"Error getting feature importances: {e}")
        return None


def plot_feature_importances(feature_importances):
    """Create a feature importance plot"""
    if feature_importances is None:
        return None

    plt.figure(figsize=(10, 6))
    top_features = feature_importances.sort_values('Importance', ascending=False).head(10)

    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()

    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_str


# Load dataset information
def get_dataset_info():
    try:
        data = pd.read_csv("synthetic_data.csv")
        features = [col for col in data.columns if col != 'target']
        return {
            'n_samples': len(data),
            'n_features': len(features),
            'feature_names': features,
            'n_classes': len(data['target'].unique())
        }
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return {
            'n_samples': 'Unknown',
            'n_features': 'Unknown',
            'feature_names': [],
            'n_classes': 'Unknown'
        }


# Load experiment results
def get_experiment_results():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("classification-synthetic-data")

        if not experiment:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.f1 DESC"]
        )

        results = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'metrics': {k: f"{v:.4f}" for k, v in run.data.metrics.items()},
                'params': {k: v for k, v in run.data.params.items() if
                           k not in ['n_samples', 'n_features', 'n_classes', 'n_informative']}
            }
            results.append(run_info)

        return results
    except Exception as e:
        print(f"Error getting experiment results: {e}")
        return []


# HTML template for home page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Classification Model Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 5px;
        }
        h1, h2, h3 {
            color: #0066cc;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: white;
        }
        .feature-input {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .input-group {
            flex: 1 0 150px;
            margin-bottom: 10px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #e9ecef;
            border: none;
            cursor: pointer;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }
        .tab.active {
            background-color: #0066cc;
            color: white;
        }
        .tab-content {
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 0 5px 5px 5px;
        }
        .tab-content.active {
            display: block;
        }
        img {
            max-width: 100%;
        }
        .model-comparison {
            overflow-x: auto;
        }
        .easter-egg {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px;
            background-color: #0066cc;
            color: white;
            border-radius: 5px;
            cursor: pointer;
            z-index: 1000;
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
            40% {transform: translateY(-20px);}
            60% {transform: translateY(-10px);}
        }
        .bounce {
            animation: bounce 2s infinite;
        }
        .konami-active {
            background: linear-gradient(124deg, #ff2400, #e81d1d, #e8b71d, #e3e81d, #1de840, #1ddde8, #2b1de8, #dd00f3, #dd00f3);
            background-size: 1800% 1800%;
            animation: rainbow 10s ease infinite;
        }
        @keyframes rainbow { 
            0%{background-position:0% 82%}
            50%{background-position:100% 19%}
            100%{background-position:0% 82%}
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MLflow Classification Model Predictor</h1>

        <div class="tabs">
            <button class="tab active" onclick="openTab(event, 'predict-tab')">Predict</button>
            <button class="tab" onclick="openTab(event, 'model-info-tab')">Model Information</button>
            <button class="tab" onclick="openTab(event, 'model-comparison-tab')">Model Comparison</button>
        </div>

        <div id="predict-tab" class="tab-content active">
            <div class="card">
                <h2>Enter Feature Values for Prediction</h2>
                <form id="prediction-form">
                   <div class="feature-input">
                        {% for feature in dataset_info['feature_names'][:10] %}
                        <div class="input-group">
                            <label for="{{ feature }}">{{ feature }}</label>
                            <input type="number" id="{{ feature }}" name="{{ feature }}" step="0.01" required value="0">
                        </div>
                        {% endfor %}
                    </div>
                    
                    {% if dataset_info['feature_names']|length > 10 %}
                    <div id="more-features" class="hidden">
                        <div class="feature-input">
                            {% for feature in dataset_info['feature_names'][10:] %}
                            <div class="input-group">
                                <label for="{{ feature }}">{{ feature }}</label>
                                <input type="number" id="{{ feature }}" name="{{ feature }}" step="0.01" required value="0">
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    
                    <button type="button" id="toggle-features" onclick="toggleFeatures()">Show More Features</button>
                    {% endif %}
                    
                    <button type="submit">Predict</button>
                </form>
                
                <div id="prediction-result" class="result hidden"></div>
            </div>
        </div>
        
        <div id="model-info-tab" class="tab-content">
            <div class="card">
                <h2>Dataset Information</h2>
                <p><strong>Number of Samples:</strong> {{ dataset_info['n_samples'] }}</p>
                <p><strong>Number of Features:</strong> {{ dataset_info['n_features'] }}</p>
                <p><strong>Number of Classes:</strong> {{ dataset_info['n_classes'] }}</p>
            </div>
            
            <div class="card">
                <h2>Feature Importances</h2>
                {% if feature_importance_img %}
                <img src="data:image/png;base64,{{ feature_importance_img }}" alt="Feature Importances">
                {% else %}
                <p>Feature importance information not available.</p>
                {% endif %}
            </div>
        </div>
        
        <div id="model-comparison-tab" class="tab-content">
            <div class="card model-comparison">
                <h2>Model Performance Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            {% if dataset_info['n_classes'] == 2 %}
                            <th>ROC AUC</th>
                            {% endif %}
                            <th>Parameters</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in experiment_results %}
                        <tr>
                            <td>{{ result['run_name'] }}</td>
                            <td>{{ result['metrics'].get('accuracy', 'N/A') }}</td>
                            <td>{{ result['metrics'].get('precision', 'N/A') }}</td>
                            <td>{{ result['metrics'].get('recall', 'N/A') }}</td>
                            <td>{{ result['metrics'].get('f1', 'N/A') }}</td>
                            {% if dataset_info['n_classes'] == 2 %}
                            <td>{{ result['metrics'].get('roc_auc', 'N/A') }}</td>
                            {% endif %}
                            <td>
                                {% for key, value in result['params'].items() %}
                                <strong>{{ key }}:</strong> {{ value }}<br>
                                {% endfor %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div id="easter-egg" class="easter-egg">🎮</div>

    <script>
        // Tab functionality
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }
            
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
        }
        
        // Toggle more features
        function toggleFeatures() {
            const moreFeatures = document.getElementById('more-features');
            const toggleBtn = document.getElementById('toggle-features');
            
            if (moreFeatures.classList.contains('hidden')) {
                moreFeatures.classList.remove('hidden');
                toggleBtn.textContent = 'Show Fewer Features';
            } else {
                moreFeatures.classList.add('hidden');
                toggleBtn.textContent = 'Show More Features';
            }
        }
        
        // Form submission
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const featureData = {};
            
            for (const [key, value] of formData.entries()) {
                featureData[key] = parseFloat(value);
            }
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(featureData),
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('prediction-result');
                resultDiv.classList.remove('hidden', 'success', 'error');
                
                if (data.error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>Error</h3><p>${data.error}</p>`;
                } else {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <h3>Prediction Result</h3>
                        <p><strong>Predicted Class:</strong> ${data.prediction}</p>
                        ${data.probabilities ? `<p><strong>Probabilities:</strong> ${JSON.stringify(data.probabilities)}</p>` : ''}
                    `;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                const resultDiv = document.getElementById('prediction-result');
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `<h3>Error</h3><p>Failed to get prediction: ${error.message}</p>`;
                resultDiv.classList.remove('hidden');
            });
        });
        
        // Easter egg - Konami code
        const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
        let konamiPosition = 0;
        
        document.addEventListener('keydown', function(e) {
            const key = e.key;
            
            // Check if the key matches the konami code at current position
            if (key === konamiCode[konamiPosition]) {
                konamiPosition++;
                
                // If completed the konami code
                if (konamiPosition === konamiCode.length) {
                    activateEasterEgg();
                    konamiPosition = 0;
                }
            } else {
                konamiPosition = 0;
            }
        });
        
        document.getElementById('easter-egg').addEventListener('click', function() {
            activateEasterEgg();
        });
        
        function activateEasterEgg() {
            document.body.classList.toggle('konami-active');
            alert('🎮 Easter Egg Activated: You found the secret! This is a tribute to the classic Konami code. 🎮');
            
            // Make the egg bounce
            const egg = document.getElementById('easter-egg');
            egg.classList.add('bounce');
            setTimeout(() => {
                egg.classList.remove('bounce');
            }, 2000);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    # Get information for the template
    model = load_best_model()
    feature_importances = get_feature_importances()
    feature_importance_img = plot_feature_importances(feature_importances)
    dataset_info = get_dataset_info()
    experiment_results = get_experiment_results()

    # Render template
    return render_template_string(
        HTML_TEMPLATE,
        dataset_info=dataset_info,
        feature_importance_img=feature_importance_img,
        experiment_results=experiment_results
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature values from request
        feature_values = request.json

        # Load model
        model = load_best_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Please make sure you have run the training script first.'})

        # Convert input to DataFrame with correct feature names
        dataset_info = get_dataset_info()
        features = dataset_info['feature_names']

        # Check if all features are provided
        missing_features = [f for f in features if f not in feature_values]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'})

        # Create DataFrame with only the required features
        input_df = pd.DataFrame([{f: float(feature_values[f]) for f in features}])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Try to get prediction probabilities if available
        try:
            probabilities = model.predict_proba(input_df)[0].tolist()
            class_probs = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        except:
            class_probs = None

        return jsonify({
            'prediction': int(prediction),
            'probabilities': class_probs
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists("best_run_id.txt"):
        print("Warning: No best model found. Please run the training script first.")

    app.run(debug=True, host='0.0.0.0', port=5000)