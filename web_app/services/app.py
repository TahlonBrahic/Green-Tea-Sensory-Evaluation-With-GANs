from flask import Flask, render_template
import os

TEMPLATE_DIR = os.path.abspath('web_app/templates')
STATIC_DIR = os.path.abspath('web_app/static/styles')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

print(TEMPLATE_DIR)

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