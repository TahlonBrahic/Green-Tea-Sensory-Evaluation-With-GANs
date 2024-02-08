from flask import Flask, render_template
import os

APP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(app_root_path, '../templates')
STATIC_DIR = os.path.join(app_root_path, '../static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/projects')
def projects():
    pass

@app.route('/tea-catechins')
def tea_catechins():
    pass

if __name__ == '__main__':
    app.run()