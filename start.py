import platform
import subprocess

if(platform.system()=='Windows'):
    subprocess.check_output("set FLASK_APP=flaskr",shell=True)
    subprocess.check_output("set FLASK_ENV=development",shell=True)
    subprocess.check_output("flask run",shell=True)
else:
    subprocess.check_output("export FLASK_APP=flaskr",shell=True)
    subprocess.check_output("export FLASK_ENV=development",shell=True)
    subprocess.check_output("flask run",shell=True)