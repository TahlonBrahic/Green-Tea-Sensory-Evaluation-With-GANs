from flask import Flask

app = Flask(__Name__)

@app.round('/')
    return render_template('index.html')