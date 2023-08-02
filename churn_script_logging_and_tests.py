'''
Testing for Churn_library python script file
Author: Nedal Altiti
Date 02/08/23

'''

import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def path():
    """
    Fixture - The test function test_import() will
    use the return of path() as an argument
    """
    return "./data/bank_data.csv"


@pytest.fixture(scope="module")
def dataframe(input_path):
    """
    Fixture - The test functions will
    use the return of dataframe() as an argument
    """
    return cls.import_data(input_path)


@pytest.fixture(scope="module",
                params=[['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category',
                         'Card_Category'],
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category'],
                        ])
def encoder_params(request):
    """
    Fixture - The test function test_encoder_helper will
    use the parameters returned by encoder_params() as arguments
    """
    cat_features = request.param
    # get dataset from pytest Namespace
    data = pytest.df.copy()
    return data, cat_features


@pytest.fixture(scope="module")
def input_train():
    """
    Fixture - The test function test_train_models will
    use the datasets returned by input_train() as arguments
    """
    # get dataset from pytest Namespace
    data = pytest.df.copy()
    return cls.perform_feature_engineering(data)


@pytest.mark.parametrize("filename",
                         ["./data/bank_data.csv",
                          "./data/no_file.csv"])
######################### UNIT TESTS ##################################
def test_import(filename):
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        data = cls.import_data(filename)
        logging.info("Testing import_data from file: %s - SUCCESS", filename)

        # store dataframe into pytest namespace for re-use in other test
        # functions
        pytest.df = data

    except FileNotFoundError:
        logging.error(
            "Testing import_data from file: %s: The file wasn't found",
            filename)

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info("Returned dataframe with shape: %s", data.shape)
    except AssertionError as err:
        logging.error("The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    # get dataset from pytest Namespace
    data = pytest.df

    try:
        cls.perform_eda(data)
        logging.info("Testing perform_eda - SUCCESS")

    except Exception as err:
        logging.error("Testing perform_eda failed - Error type %s", type(err))


def test_encoder_helper(params):
    '''
    test encoder helper
    '''
    data, cat_features = params

    try:
        newdf = cls.encoder_helper(data, cat_features)
        logging.info("Testing encoder_helper with %s - SUCCESS", cat_features)

    except Exception as err:
        logging.error(
            "Testing encoder_helper failed - Error type %s",
            type(err))

    try:
        assert newdf.select_dtypes(include='object').columns.tolist() == []
        logging.info("All categorical columns were encoded")
    except AssertionError:
        logging.error(
            "At least one categorical columns was NOT encoded" +
            "- Check categorical features submitted")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        data = pytest.df
        X_train_test, X_test_test, y_train_test, y_test_test = cls.perform_feature_engineering(
            data)
        logging.info("Testing perform_feature_engineering - SUCCESS")

    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering failed - Error type %s",
            type(err))

    try:
        assert X_train_test.shape[0] > 0
        assert X_train_test.shape[1] > 0
        assert X_test_test.shape[0] > 0
        assert X_test_test.shape[1] > 0
        assert y_train_test.shape[0] > 0
        assert y_test_test.shape[0] > 0
        logging.info(
            "perform_feature_engineering returned Train / Test set of shape %s %s",
            X_train_test.shape,
            X_test_test.shape)

    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering failed - Error type: %s",
            type(err))


def test_train_models(input_train_param):
    '''
    test train_models
    '''
    try:
        cls.train_models(*input_train_param)
        logging.info(
            "Testing train_models with input: %s - SUCCESS",
            input_train_param)
    except Exception as err:
        logging.error(
            "Testing train_models with input: %s - Error type: %s",
            input_train_param,
            type(err))
