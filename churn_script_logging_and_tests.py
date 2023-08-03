'''
Testing for Churn_library python script file
Author: Nedal Altiti
Date 02/08/23

'''
import os
import logging
from math import ceil
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = cls.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(df=df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error('Perform EDA function error')
        raise err

    # Assert if `churn_dist.png` is created
    try:
        assert os.path.isfile("./images/eda/churn_dist.png") is True
        logging.info('File %s was found', 'churn_dist.png')
    except AssertionError as err:
        logging.error('File %s not found', 'churn_distribution.png')
        raise err

    # Assert if `customer_age_dist.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/customer_age_dist.png") is True
        logging.info('File %s was found', 'customer_age_dist.png')
    except AssertionError as err:
        logging.error('File %s not found', 'customer_age_dist.png')
        raise err

    # Assert if `marital_status_dist.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/marital_status_dist.png") is True
        logging.info('File %s was found', 'marital_status_dist.png')
    except AssertionError as err:
        logging.error('File %s not found', 'marital_status_dist.png')
        raise err

    # Assert if `total_transaction_dist.png` is created
    try:
        assert os.path.isfile(
            "./images/eda/total_transaction_dist.png") is True
        logging.info('File %s was found', 'total_transaction_dist.png')
    except AssertionError as err:
        logging.error(
            'File %s not found',
            'total_transaction_dist.png')
        raise err

    # Assert if `heatmap.png` is created
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error('File %s not found', 'heatmap.png')
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    # Load DataFrame
    dataframe = cls.import_data("./data/bank_data.csv")

    # Categorical Features
    cat_columns = [
        'Gender', 'Education_Level',
        'Marital_Status', 'Income_Category',
        'Card_Category']
    try:
        _ = cls.encoder_helper(
            df=dataframe,
            category_lst=cat_columns,
            response="Churn"
        )
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: ERROR")

        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
# Load the DataFrame
    dataframe = cls.import_data("./data/bank_data.csv")
    try:
        (_, x_test, _, _) = cls.perform_feature_engineering(
            df=dataframe,
            response='Churn'
        )
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except KeyError as err:
        logging.error(
            'The `Churn` column is not present in the DataFrame: ERROR')
        raise err

    try:
        # x_test size should be 30% of `data_frame`
        assert (
            x_test.shape[0] == ceil(
                dataframe.shape[0] *
                0.3)) is True   # pylint: disable=E1101
        logging.info(
            'Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR')
        raise err


def test_train_models():
    '''
    test train_models
    '''
    # Load the DataFrame
    dataframe = cls.import_data("./data/bank_data.csv")

    # Feature engineering
    (x_train, x_test, y_train, y_test) = cls.perform_feature_engineering(
        df=dataframe,
        response='Churn')

    # Assert if `logistic_model.pkl` file is present
    try:
        cls.train_models(x_train, x_test, y_train, y_test)
        logging.info('Testing train_models: SUCCESS')
    except AssertionError as err:
        logging.error('Testing train_models: ERROR')
        raise err

    # Assert if `rfc_model.pkl` file is present
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('File %s was found', 'rfc_model.pkl')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if `ROC_curves.png` file is present
    try:
        assert os.path.isfile('./images/results/ROC_curves.png') is True
        logging.info('File %s was found', 'ROC_curves.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if `confusion_matrix_rf.png` file is present
    try:
        assert os.path.isfile('./images/results/confusion_matrix_rf.png') is True
        logging.info('File %s was found', 'confusion_matrix_rf.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if `confusion_matrix_lr.png` file is present
    try:
        assert os.path.isfile('./images/results/confusion_matrix_lr.png') is True
        logging.info('File %s was found', 'confusion_matrix_lr.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    # Assert if `feature_importances_Random_Forest.png` file is present
    try:
        assert os.path.isfile(
            './images/results/feature_importances_Random_Forest.png') is True
        logging.info('File %s was found', 'feature_importances_Random_Forest.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()