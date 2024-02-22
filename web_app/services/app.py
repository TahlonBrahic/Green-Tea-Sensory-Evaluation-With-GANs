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

    # Plot Related 
    plot = create_tea_catechins_plot()s
    plot_div = plotly.offline.plot(plot, output_type='div', include_plotlyjs=False)
    load_tea_catechin_models() # Lazy loading, models aren't loaded until you visit this exact page to avoid website lag

    # Model Selection Releated
    model_names = ['Random Forest', 'Multilayer Perceptron' 'Recurrent Neural Network']

    if request.method == 'POST':
        data = request.json

        model_choice = data.get('model_choice') # Need to implement dropdown and relevant information

        # Extract features from the request
        features = [
            data.get(f'feature{i}') for i in range(1, 10)  
        ]

        prediction = predict(model_choice, features)

        return jsonify({'prediction': prediction})

    # For GET requests, serve the project page
    return render_template('tea_catechins.html', plot_div=plot_div)  # Assuming you have a template for this page


if __name__ == '__main__':
    app.run(debug=True)