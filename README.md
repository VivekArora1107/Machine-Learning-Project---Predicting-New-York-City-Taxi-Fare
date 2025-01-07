# New York City Taxi Fare Prediction

## Description
In this project, we aim to predict the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. This project is part of a playground competition hosted in partnership with Google Cloud and Coursera. The challenge is to improve upon a basic estimate based on the distance between two points using Machine Learning techniques.

## Project Outline
1. Download the dataset
2. Explore and Analyze the data
3. Prepare the dataset for training ML models
4. Train baseline models
5. Make predictions and submit to Kaggle
6. Perform feature engineering
7. Train and evaluate different models
8. Tune hyperparameters for the best model
9. Publish the Project online

## Dependencies
- Python 3.12.4
- opendatasets
- pandas
- numpy
- plotly
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Installation
To install the required dependencies, run:
```bash
pip install opendatasets pandas numpy plotly matplotlib seaborn scikit-learn xgboost
```

## Dataset
The dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/overview).

## Data Exploration
We start by exploring the dataset to understand its structure and contents. This includes checking the number of rows, columns, and data types.

## Data Preparation
We prepare the dataset for training by:
- Splitting the data into training and validation sets
- Filling or removing missing values
- Extracting input and output columns

## Feature Engineering
We perform feature engineering by:
- Extracting parts of the date (Year, Month, Day, Weekday, Hour)
- Removing outliers and invalid data
- Adding distance between pickup and dropoff locations
- Adding distance from popular landmarks

## Model Training
We train several models including:
- Mean Regressor
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

## Hyperparameter Tuning
We tune the hyperparameters for the XGBoost model to improve its performance.

## Results
The final model achieves a validation RMSE of approximately $3.28, placing it among the top 30% on the Kaggle leaderboard.

## Usage
To run the notebook, execute the cells in order. The notebook includes code for downloading the dataset, data exploration, data preparation, feature engineering, model training, and hyperparameter tuning.

## License
This project is licensed under the MIT License.
