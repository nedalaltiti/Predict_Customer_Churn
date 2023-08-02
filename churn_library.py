# library doc string
'''
Churn Customer Prediction
Author: Nedal Altiti
Date: 01 / 08 / 23
'''

# import libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth, index_col=0)

    # Encode Churn (The target) : 0 = didn't churn ; 1 = Churned
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1, inplace=True)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    eda_df = df.copy(deep=True)

    # Analyze Categorical Features and plot distribution
    cat_columns = df.select_dtypes(include='object').columns.tolist()
    for cat_column in cat_columns:
        plt.figure(figsize=(7, 4))
        (df[cat_column].value_counts(normalize=True).plot(
            kind='bar', rot=45, title=f'{cat_column} - %Churn'))
        plt.savefig(os.path.join('images/eda',
                                 f'{cat_column}.png'),
                    bbox_inches='tight')
        plt.close()

    # Analyze Numerical Features

    # Total Transaction Count Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(eda_df['Total_Trans_Ct'], kde=True)
    plt.title('Distribution of Total Transaction Count')
    plt.savefig(fname='./images/eda/total_transaction_dist.png')
    plt.close()

    # churn distributions
    plt.figure(figsize=(10, 5))
    eda_df['Churn'].plot(kind='hist', title='Churned vs Unchurned Customers')
    plt.savefig(fname='./images/eda/churn_dist.png')
    plt.close()

    # Customer Age Distribution
    plt.figure(figsize=(10, 5))
    eda_df['Customer_Age'].plot(kind='hist',
                                title='Distribution of Customers Age')
    plt.savefig(fname='./images/eda/customer_age_dist.png')
    plt.close()

    # Heatmap
    numerical_columns = eda_df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        eda_df[numerical_columns].corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')
    plt.close()

    # Correlation between Transaction Amount (independent) and Transaction
    # Count (dependent)
    plt.figure(figsize=(15, 7))
    (df[['Total_Trans_Amt', 'Total_Trans_Ct']]
        .plot(x='Total_Trans_Amt',
              y='Total_Trans_Ct',
              kind='scatter',
              title='Total_Trans_Amt - Total_Trans_Ct')
     )
    plt.savefig(fname='./images/eda/corr_trans.png')
    plt.close()

    return eda_df


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                        used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    encoder_df = df.copy(deep=True)

    for category in category_lst:
        category_groups = df[response].groupby(df[category]).mean()
        new_feature = category + '_' + response
        encoder_df[new_feature] = df[category].map(category_groups)

    encoder_df.drop(category_lst, axis=1, inplace=True)
    return encoder_df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
                                    used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Categorical Features
    cat_columns = df.select_dtypes(include='object').columns.tolist()
    # Encode categorical features using mean of response variable on category
    df = encoder_helper(df, cat_columns, response='Churn')

    # Alternative way for encoding is to use one-hot encoder
    # One-Hot Encode Categorical Features
    # encoder = OneHotEncoder(sparse=False)
    # encoded_features = encoder.fit_transform(df[cat_columns])
    # Create a new dataframe with encoded features
    # encoded_df = pd.DataFrame(encoded_features,
    #               columns=encoder.get_feature_names_out(cat_columns))
    # Concatenate the encoded features with the numerical features
    # df_encoded = pd.concat([df.drop(cat_columns, axis=1), encoded_df], axis=1)

    y = df[response]
    X = df.drop(response, axis=1)

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    train_report_lr = classification_report(y_train, y_train_preds_lr)
    train_report_rf = classification_report(y_train, y_train_preds_rf)
    test_report_lr = classification_report(y_test, y_test_preds_lr)
    test_report_rf = classification_report(y_test, y_test_preds_rf)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].text(0, 0, train_report_lr)
    axes[0, 0].set_title("Training Report - Logistic Regression")
    axes[0, 0].axis('off')
    plt.savefig(
        os.path.join(
            "./images/results",
            'classification_report.png'),
        bbox_inches='tight')
    plt.close()

    return train_report_lr, train_report_rf, test_report_lr, test_report_rf


def feature_importance_plot(model, X_data, model_name, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             important features
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(output_pth, fig_name))
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              ROC curve for rf and lr models
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    # Plot ROC curve for Random Forest
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)

    # Plot ROC curve for Logistic Regression
    RocCurveDisplay.from_estimator(
        lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(
        os.path.join(
            "./images/results",
            'Roc_curves.png'),
        bbox_inches='tight')
    plt.close()

# save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Display the confusion matrix for the Random Forest model
    # Confusion matrix for Random Forest model
    cm_rf_train = confusion_matrix(y_train, y_train_preds_rf)
    cm_rf_test = confusion_matrix(y_test, y_test_preds_rf)

    # Confusion matrix for Logistic Regression model
    cm_lr_train = confusion_matrix(y_train, y_train_preds_lr)
    cm_lr_test = confusion_matrix(y_test, y_test_preds_lr)

    # Print the confusion matrices
    print("Confusion Matrix - Random Forest (Train):")
    print(cm_rf_train)
    print("Confusion Matrix - Random Forest (Test):")
    print(cm_rf_test)
    print("Confusion Matrix - Logistic Regression (Train):")
    print(cm_lr_train)
    print("Confusion Matrix - Logistic Regression (Test):")
    print(cm_lr_test)

    # Display feature importance on train data
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_train,
                            'Random_Forest',
                            "./images/results")

    def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
        '''
    plot confusionmatrix for rf and lr models on test dataset
    input:
              confusion matrix
              classes : Churn or Not Churn
    output:
              Confusion Matrix plot for both rf and lr models
    '''
        plt.imshow(cm, interpolation='nearest', cmap='viridis')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if confusion_matrix.max() < 1 else 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        # Plot confusion matrix for Random Forest (Train)
    plt.figure()
    plot_confusion_matrix(
        cm_rf_train,
        classes=[
            'Not Churn',
            'Churn'],
        title='Confusion Matrix - Random Forest (Train)')
    plt.close()

    # Plot confusion matrix for Random Forest (Test)
    plt.figure()
    plot_confusion_matrix(
        cm_rf_test,
        classes=[
            'Not Churn',
            'Churn'],
        title='Confusion Matrix - Random Forest (Test)')
    plt.savefig(
        os.path.join(
            "./images/results",
            'confusion_matrix_rf.png'),
        bbox_inches='tight')
    plt.close()

    # Plot confusion matrix for Logistic Regression (Train)
    plt.figure()
    plot_confusion_matrix(
        cm_lr_train,
        classes=[
            'Not Churn',
            'Churn'],
        title='Confusion Matrix - Logistic Regression (Train)')
    plt.close()

    # Plot confusion matrix for Logistic Regression (Test)
    plt.figure()
    plot_confusion_matrix(
        cm_lr_test,
        classes=[
            'Not Churn',
            'Churn'],
        title='Confusion Matrix - Logistic Regression (Test)')
    plt.savefig(
        os.path.join(
            "./images/results",
            'confusion_matrix_lr.png'),
        bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    dataset = import_data("./data/bank_data.csv")
    print('Dataset loaded successfully. Initiating data exploration...')
    perform_eda(dataset)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataset, response='Churn')
    print('Feature engineering completed. Starting model training...')
    train_models(X_train, X_test, y_train, y_test)
    print('Model training completed. Best model weights and performance metrics saved.')
