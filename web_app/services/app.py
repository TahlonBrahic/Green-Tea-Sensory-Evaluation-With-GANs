from flask import Flask, render_template, request, jsonify
import os
import sys
import plotly
import json

# Define application's root directory path
APP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(APP_ROOT_PATH, '../templates')
STATIC_DIR = os.path.join(APP_ROOT_PATH, '../static')

# Add to Python's module search path 
models_path = os.path.abspath(os.path.join(APP_ROOT_PATH, '../static/models'))
if models_path not in sys.path:
    sys.path.append(models_path)

from tea_catechins_model_manager import * # Not optimal but the variables are so specific the namespace is clear

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route('/')
@app.route('/#')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/projects/tea-catechins', methods=['GET', 'POST']) 
def tea_catechins():

    load_tea_catechin_models()

    model_names = ['Random Forest', 'Multilayer Perceptron', 'Recurrent Neural Network']
    model_choice = 'Random Forest'  # Default model choice
    
    if request.method == 'POST':
        try:
            data = request.json
            model_choice = data.get('model_choice', 'Random Forest')

            # Extract features from the request, with error handling for missing data
            features_keys = [
                'Catechin', 'Epicatechin', 'Gallocatechin', 'Epigallocatechin',
                'Catechin_Gallate', 'Epicatechin_Gallate', 'Gallocatechin_Gallate',
                'Epigallocatechin_Gallate', 'Caffeine'
            ]
            features = [data.get(key, 0) for key in features_keys]  # Default to 0 for missing keys

            prediction = predict(model_choice, features)
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # For GET requests or if POST fails, generate and return the default plot
    try:
        plot = create_tea_catechins_plot(model_name=model_choice)
        plot_div = plotly.offline.plot(plot, output_type='div', include_plotlyjs=True)
    except Exception as e:
        return f"Error generating plot: {str(e)}", 500

    return render_template('tea_catechins.html', plot_div=plot_div, model_names=model_names)

@app.route('/get-new-graph-data', methods=['GET'])
def get_new_graph_data():
    try:
        model_choice = request.args.get('model_choice', 'Random Forest')
        plot = create_tea_catechins_plot(model_name=model_choice)
        plot_json = plot.to_json()
        return plot_json
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)