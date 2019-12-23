<<<<<<< HEAD

from flask import Flask, render_template, request, url_for, redirect, session

#--------------------------------------------------------------------------------------------------
# Global
#--------------------------------------------------------------------------------------------------
# Flask App
app = Flask(__name__)


#--------------------------------------------------------------------------------------------------
# Route: Index
#--------------------------------------------------------------------------------------------------
@app.route('/index')
@app.route('/')
def index():
    
    print("Hello World")

#--------------------------------------------------------------------------------------------------
# MAIN
#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #app_dash.run_server(debug = True)
    app.run(debug = True) 
=======
import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Data Viz Project!'

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
>>>>>>> 8217244e88f4d5e4edb298ada79b98bcbbf14c5d
