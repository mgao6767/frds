import os
from .data import wrds
from .ra import ra

if __name__ == "__main__":
    print('run!')
    usr = os.getenv('WRDS_USRNAME')
    pwd = os.getenv("WRDS_PASSWORD")
    conn = wrds.Connection(usr, pwd)
    libs = conn.list_libraries()
    from pprint import pprint
    pprint(libs)
    pprint(conn.list_tables('crsp'))
    pprint(conn.describe_table('crsp', 'crsp_daily_data'))
    pprint(conn.get_table('crsp', 'dsi', obs=10))
