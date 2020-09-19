import psycopg2
import pandas as pd

try:
    connection = psycopg2.connect(host='localhost', dbname='sharklanddb',
                                  user='adminsharkland', port=7001, password='atomic181')
    posts_info = pd.read_sql("select * from post", connection)
    posts_info


except (psycopg2.Error) as e:
    print(e)
finally:

    connection.close()
