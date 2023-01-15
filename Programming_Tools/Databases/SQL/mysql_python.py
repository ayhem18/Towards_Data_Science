import os
from configparser import ConfigParser
from mysql.connector import Error


# hardcoding the credential is not as safe and is error-prone.
# One more reliable way it to used configuration files
def read_db_config(config_file_path: str, section:str='mysql') -> dict:
    """
    This function reads configuration file, parse the credentials and return them in a dictionary-like object.
    : param config_file_path: the path to the configuration file
    : param section: the section to consider from the configuration file
    : return  a dictionary with the connection's parameters
    """
    # initialize a parser object
    parser = ConfigParser()
    parser.read(config_file_path)
    
    # check if the section set in the argument is indeed in the configuration file (read by the parser)
    try:
        if parser.has_section(section):
            # extract the items 
            items = parser.items(section)
            # convert them to a dict
            return dict((item[0], item[1]) for item in items)

    except Error as e:
        print(e)

    # the reuslt will be of None type in case the section is not present in the configuration file
    

# let's see how the function works
# let's have a function that creates a configuration file given hard-coded credentials
def write_db_config(host: str, database: str, user: str, password:str, config_file_path:str=None, section:str='mysql'):
    if config_file_path is None:
        config_file_path = os.path.join(os.path.dirname(__file__), f'{user}_{host}_{database}')
    
    # let's first check if the file ends with the appropriate extension and add if necessary
    if not config_file_path.endswith('.ini'):
        config_file_path += '.ini'

    params = {"host": host, "database":database, "user": user, "password": password}
    # create the file
    with open(config_file_path, 'w') as f:
        f.write(f'[{section}]\n')
        for param, param_value in params.items():
            f.write(f'{param}= {param_value}\n')
    return config_file_path

from mysql.connector import MySQLConnection

# let's introduce a function that will connect via config files
def connect_config(config_file_path, section='mysql'):
    # extract the credential from the configuration file
    db_config = read_db_config(config_file_path, section=section)
    try:    
        with MySQLConnection(**db_config) as conn:
            if conn.is_connected():
                print('Connection established.')
            else:
                print('Connection failed.')

        return conn
    except Error as e:
        print(e)


if __name__ == '__main__':
    # create the file given the database credentials
    pass    