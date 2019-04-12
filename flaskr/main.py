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
        input = request.form['userInput']
        print("Input: ", input)
        print("Type: ", type(input))
        input = ast.literal_eval(input)
        # input = [n.strip() for n in input]
        print("Parsed Type: ", type(input))
        result = predict_data(input)
        print("Result: ", result)
        response = []
        for r in result:
            print(r)
            if r == 0:
                response.append("Healthy")
                
            else:
                response.append("Infected")
        data = build_json()    
        return render_template('chart/task3.html', result=response, input=input, json=data)
    else:
        return render_template('chart/task3.html')
    







