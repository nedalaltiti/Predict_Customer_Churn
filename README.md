# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
The first project of the Machine Learning DevOps Engineer Nanodegree focuses on developing production-ready clean code using industry best practices. The project's main objective is to predict customer churn for banking customers, which involves solving a classification problem.
## Project Description
The first project of the Machine Learning DevOps Engineer Nanodegree focuses on developing production-ready clean code using industry best practices. The project's main objective is to predict customer churn for banking customers (Binary Classification Task).
To achieve this, the project proposes the following approach:
•	Perform exploratory data analysis (EDA) to gain insights into the dataset and understand the patterns and relationships within the data.
•	Conduct feature engineering to transform and preprocess the data, including handling missing values, encoding categorical variables, and scaling numerical features.
•	Split the dataset into training and testing sets to evaluate the performance of the machine learning models accurately.
•	Train multiple classification models using various algorithms such as logistic regression, decision trees, random forests, or gradient boosting.
•	Evaluate the performance of the trained models using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
•	Select the best-performing model based on the evaluation results and fine-tune its hyperparameters using techniques like grid search or random search.
•	Once the final model is selected, use it to make predictions on new, unseen data to identify customers who are likely to churn.
•	Save the trained model using the joblib library and store it in a .pkl file for future use.
•	Generate visualizations and plots to analyze the results, such as ROC curves, feature importances, and other relevant insights.
•	Implement logging and testing to ensure the code's functionality and track any errors or issues that may arise during execution.
•	Document the entire project, including the purpose, methodology, code structure, and instructions for running the code and reproducing the results.

## Files and data description
churn_library.py: This is a Python module that contains the functions and classes necessary for performing customer churn analysis. It includes functions for data preprocessing, feature engineering, model training, and evaluation.

churn_script_logging_and_tests.py: This is a Python script that demonstrates how to use the functions from the churn_library.py module. It also includes logging statements to track the progress of the script and unit tests to ensure the correctness of the implemented functions.

customer_data.csv: This is a CSV file that contains the customer data used for the churn analysis. It includes information such as customer ID, account length, international plan, voice mail plan, number of voicemail messages, total day minutes, total day calls, total day charge, and more.

churn.png: This is an image file that shows a visualization of the churn rate in the customer data. It can be used to gain insights into the distribution of churned and non-churned customers.

README.md: This is a markdown file that provides a brief description of the project, including instructions on how to run the script and interpret the results.
## Running Files
I used Python version 3.10.11
The required packages are provided in the requirements.txt file
To run the project, execute the script python churn_library.py from the project folder
The pytest python package was used to test the churn_library.py project script. To execute the unit tests, you can simply run the pytest command from the main project folder in the command line. As a result of running the tests, the project functions will be automatically tested, and a log file will be generated in the logs folder.


##Model Evaluation
Random Forest vs Logistic Regression

![image](https://github.com/nedalaltiti/Predict_Customer_Churn/assets/106015333/0f0cec3f-5631-49d9-b625-bd19fdb61384)

Random Forest Confusion Matrix

![image](https://github.com/nedalaltiti/Predict_Customer_Churn/assets/106015333/66554330-35e3-445d-ad42-346bedfb37c2)

Logistic Regression Confusion Matrix 

![image](https://github.com/nedalaltiti/Predict_Customer_Churn/assets/106015333/41141256-c6ff-4be4-aea9-fb77ddc975b2)

Important Features

![image](https://github.com/nedalaltiti/Predict_Customer_Churn/assets/106015333/3ccb4569-49cf-4ea1-af66-19e6b2bec9d1)



