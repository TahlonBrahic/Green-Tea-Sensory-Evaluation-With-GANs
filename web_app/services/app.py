from flask import Flask, render_template
import os
from tea_catechin_model_manager import load_tea_catechin_models

APP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(APP_ROOT_PATH, '../templates')
STATIC_DIR = os.path.join(APP_ROOT_PATH, '../static')

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
    return render_template('tea_catechins.html')

if __name__ == '__main__':
    app.run(debug=True)