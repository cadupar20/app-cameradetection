from flask import Flask, render_template
import face_recognize
import create_data

app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html', name=name)

@app.route('/exec')
def parse(name=None):
    print("done")
    return render_template('index.html', name=name)

@app.route('/exec2')
def parse1(name=None):
    print("done")
    return render_template('index.html', name=name)

if __name__ == '__main__':
    app.debug = True
    app.run()