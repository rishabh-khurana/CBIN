import functools
import json

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from flaskr.db import get_db
from flaskr.task1 import *

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
    return render_template('chart/task2.html')
    
@bp.route('/task3', methods=['GET'])
def task3():
    return render_template('chart/task3.html')
