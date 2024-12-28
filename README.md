# stock-prediction

### Introduction: 
A Machine learning application to predict the stock prices for the National Stock Exchange of India Ltd. (NSE).
The NIFTY 50 is owned and managed by India Index Services and Products Ltd. (IISL). IISL is India's first specialized company focused on an index as a core product.

### Prerequisites:
- python 3 
- numpy for adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
- pandas to open source, data analysis and manipulation tool. 
- sklearn.tree import DecisionTreeRegressor for decision-making tool. 
- sklearn.linear_model import LinearRegression for Ordinary least squares Linear Regression.
- sklearn.model_selection import train_test_split to Split arrays or matrices into random train and test subsets.
- matplotlib.pyplot as plt for MATLAB-like plotting framework.
- plt.style.use('bmh'),in this project we are using bmh style, other style options are seaborn and ggplot

### Usage:
- import all required libraries.
- import local CSV stock data or read web data using DataReader.
- analyze the data, find the number of columns and rows, check for null values.
- select the target column and convert it into a NumPy array.
- split the data into 75% training and 25% testing.
- create and train the model.
- visualize the data

### Sources:
  - Data https://www.kaggle.com/rohanrao/nifty50-stock-market-data
  
### Reference:
- DLithe ML training https://dlithe.com/machine-learning-course-training-bangalore/
