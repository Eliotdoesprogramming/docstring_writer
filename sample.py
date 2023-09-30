import os
import psycopg2
# add documentation to this function in python, make sure to include type hints for param and return values:

def connect_db():
    host = os.environ.get('DB_HOST')
    port = os.environ.get('DB_PORT')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')
    dbname = os.environ.get('DB_NAME')
    conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)
    return conn
