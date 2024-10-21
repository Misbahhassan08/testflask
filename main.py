from flask import Flask
import os
from flask_cors import CORS


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
CORS(app)
app.app_context().push()



@app.route('/')
def hello_world():
    return "hello , world"
if __name__ == '__main__':
    app.run(port=8080, debug=True, use_reloader=False)
    #app.run(host='0.0.0.0', port=port)
