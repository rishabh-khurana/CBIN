import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from flaskr.db import get_db

bp = Blueprint('chart', __name__, url_prefix='/chart')

@bp.route('/index', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        test = request.form['test']
        text = request.form['text']
        if test == "1":
            print("Correct")
            return redirect(url_for('chart.template1', data=text))
    
    return render_template('base.html')
        
        
@bp.route('/template1', methods=['GET'])
def template1():
    data = request.args['data']
    return render_template('chart/template1.html', data=data)