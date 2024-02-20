from flask import Flask, render_template, request
import os
import sys
import plotly

APP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(APP_ROOT_PATH, '../templates')
STATIC_DIR = os.path.join(APP_ROOT_PATH, '../static')

models_path = os.path.abspath(os.path.join(APP_ROOT_PATH, '../static/models'))
if models_path not in sys.path:
    sys.path.append(models_path)

from tea_catechins_model_manager import * # I know this isn't the best but the variables are so specific the namespace is clear

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
    plot = create_tea_catechins_plot()
    plot_div = plotly.offline.plot(plot, output_type='div', include_plotlyjs=False)
    load_tea_catechin_models() # Lazy loading, models aren't loaded until you visit this exact page to avoid website lag

    if request.method == 'POST':
        data = request.json

        # Extract features from the request
        features = [
            data.get(f'feature{i}') for i in range(1, 10)  # Adjust range if feature numbering is different
        ]

        # Prepare input to the model (adjust according to your model's requirements)
        # This is a generic way; you might need to adjust it based on how your model expects the input
        input_data = {Model_RF_session.get_inputs()[0].name: [features]}  # Assuming all models expect the same input format

        # Make predictions
        predictions = {
            'model1_prediction': Model_RF_session.run(None, input_data)[0].tolist(),
            'model2_prediction': Model_MLP_session.run(None, input_data)[0].tolist(),
            'model3_prediction': Model_RNN_session.run(None, input_data)[0].tolist()
        }

        # Return predictions as JSON
        return jsonify(predictions)
    
    # For GET requests, serve the project page
    return render_template('tea_catechins.html', plot_div=plot_div)  # Assuming you have a template for this page


if __name__ == '__main__':
    app.run(debug=True)