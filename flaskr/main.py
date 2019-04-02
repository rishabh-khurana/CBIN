import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from flaskr.db import get_db

bp = Blueprint('main', __name__, url_prefix='/main')
    
@bp.route('/task1', methods=['GET'])
def task1():
    return render_template('chart/task1.html')
    
@bp.route('/task2', methods=['GET'])
def task2():
    return render_template('chart/task1.html')
    
@bp.route('/task3', methods=['GET'])
def task3():
    return render_template('chart/task1.html')
