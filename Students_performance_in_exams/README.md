# Students performance in _exams
[Data source](https://www.kaggle.com/datasets/whenamancodes/students-performance-in-exams)
***

The goal of this project is to predict test score in 3 types of tests. Its purpose is to learn and practice the usage of regression algorithms.


### Technologies
***
- Python 3.10.4
- Pandas 1.4.2
- Numpy 1.22.3
- Scikit-Learn 1.1.2
- XGBoost 1.6.2
- Seaborn 0.11.2
- Matplotlib 3.5.1

<br>

### About the data
***
This dataset contains data about 1000 students including 5 features and 3 target variables defining scores in different types of tests: math, writing and reading.

<br>

### Data Cleaning
***
- Renamed columns
- One Hot Encoding
- Removed some dependant variables after One Hot Encoding

<br>

### Models created
***
#### Evaluation
Each created model was evaluated using MSE and R2 score, but mainly focused on MSE.

<br>

#### Basic model
There were created 6 basic models: Linear regression, Decision tree regressor, Random forest regressor, XGBoost regressor, SVM regressor,  Neighbors regressor, all ran for 3 target variables. 2 best models were chosen for hyperparameter tuning and validating.

<br>

### Conclusion
***
Best choice of model for math score was SVM with MSE = 187.94 on test and for writing and reading score best was linear regressor with MSE equal to 182.94 and 174.48 respectevily. Feature importance for both models points out to whether lunch was reduced or free.
