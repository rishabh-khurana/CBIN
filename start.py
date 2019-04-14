import platform
import os

if platform.system() == 'Windows':
    os.system("set FLASK_APP=flaskr & set FLASK_ENV=development & flask run")
else:
    os.system("export FLASK_APP=flaskr & export FLASK_ENV=development & python -m flask run")