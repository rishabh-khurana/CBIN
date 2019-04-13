import functools
import json
import ast

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from flaskr.db import get_db
from flaskr.task1 import *
from flaskr.task2 import *
from flaskr.task3 import *

bp = Blueprint('main', __name__, url_prefix='/main')
    
@bp.route('/task1', methods=['GET'])
def task1():
    data = stacked_json()
    title = []
    for d in data:
        tmp = d['title']['text']
        tmp = tmp.split("grouped", 1)[0]
        title.append(tmp)
        
    return render_template('chart/task1.html', json=data, title=title)
    
@bp.route('/task2', methods=['GET'])
def task2():
    data = treemap_json()
    return render_template('chart/task2.html', json=data)
    
@bp.route('/task3', methods=['GET', 'POST'])
def task3():
    if request.method == 'POST':
        input = []
        concat = []
        print(request.form)
        for item in request.form:
            concat.append(float(request.form[item]))   
        # print("Parsed Type: ", type(input))
        input.append(concat)
        result = predict_data(input)
        if result == 0: response = "Healthy"
        else: response = "Infected"
        data = build_json()    
        # return render_template('chart/task3.html')
        return render_template('chart/task3.html', result=response, json=data)
    else:
        return render_template('chart/task3.html')
    







