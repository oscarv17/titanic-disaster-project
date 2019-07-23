import os
import kaggle
from dotenv import find_dotenv, load_dotenv
import logging

# setting credentials
os.system('set KAGGLE_USERNAME =' + os.environ.get('kaggle_username'))
os.system('set KAGGLE_KEY =' + os.environ.get('kaggle_key'))


# function to extract the data
def extractData(path):  
    os.system('kaggle datasets download mhouellemont/titanic -f train.csv -p %s'%path)
    os.system('kaggle datasets download mhouellemont/titanic -f test.csv -p %s'%path)
    
def main(project_dir):
    logger = logging.getLogger(__name__)
    logger.info('Getting raw data')
    
    # set data paths
    raw_data_path = os.path.join(os.path.pardir,'data','raw')

    # extract the data
    extractData(raw_data_path)
    logger.info('Data downloaded')

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__),os.pardir,os.pardir)
    
    # set up logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format)
    
    # loading dotenv
    dotenv = find_dotenv()
    load_dotenv(dotenv)
    
    main(project_dir)
