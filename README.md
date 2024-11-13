
# Credit Scoring Model for Client Risk Assessment

## Project Description
This project aims to predict the probability of a client's default based on the behavior of previous clients. The solution seeks to provide **transparency** in financial decision-making, allowing professionals at financial institutions to better interpret credit decisions.

## Proposed Solution
- **Scoring Model**: A machine learning model to predict the likelihood of a client defaulting.
- **Interactive Dashboard**: A **Streamlit** dashboard that enables financial professionals to easily visualize and interpret the model's predictions.

## Plan and Approach
### I) Data Preprocessing
- **Data Cleaning**: Removal and imputation of missing values.
- **Merging Datasets**: Combined data from multiple CSV files.
- **Class Balancing**: Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance class distribution.

### II) Feature Reduction
- **VarianceThreshold**: Removed constant features.
- **Correlation**: Removed highly correlated features.
- **Information Gain (Mutual Information)**: Measured feature informativeness and removed low-variance features.
- **Variance Inflation Factor (VIF)**: Evaluated multicollinearity among features.
- **Permutation Importance**: Assessed the importance of features for model predictions.

### III) Modeling
- **Classification Algorithms Tested**:
  1. **Logistic Regression**
  2. **Random Forest**
  3. **XGBoost**
  4. **SVC (Support Vector Classification)**
  5. **LightGBM**
  
  **LightGBM** was selected as the most effective model for prediction.

- **Evaluation**: Used confusion matrix, ROC-AUC curve, and optimized the classification probability threshold.
- **Hyperparameter Tuning**: Applied **GridSearch** to find optimal hyperparameters.

### IV) Model Interpretability
- **Global Interpretability**: Calculated the importance of features used by the model.
- **Local Interpretability**: Explained predictions for individual clients.

### V) Model Deployment
- **Dashboard**: Built an interactive dashboard using **Streamlit** to visualize predictions.
- **Flask API**: Deployed an API to serve the model.
- **Heroku**: Hosted the application on Heroku for cloud deployment.
- **GitHub**: Used GitHub for version control and project management.

## Dataset
The dataset consists of multiple CSV files containing behavioral information for 100,000 clients.

- **Feature Engineering**: Created new features using aggregations like `mean`, `max`, `min`, `sum`, or `var`.
- **Handling Missing Values**: Managed missing data through deletion and imputation where necessary.
- **Categorical Variables**: Applied **One Hot Encoding** for categorical variables.
- **Standardization**: Used **Standard Scaler** to normalize data.

## Evaluation Metrics
- **Confusion Matrix**: Measured model performance on positive and negative classes.
- **ROC-AUC Curve**: Assessed model performance across all classes.
- **Banking Cost Reduction**: Calculated cost variations based on classification probability thresholds.
- **AUC and F-beta Score Optimization**: Optimized metrics by adjusting probability thresholds and parameters.

## Libraries & Tools Used
- **Python** 
- **pandas**, **scikit-learn**, **XGBoost**, **LightGBM**, **Matplotlib**, **Seaborn**
- **Streamlit** for dashboard
- **Flask** for API
- **Heroku** for deployment
- **A dahsboard is accessible via this link : https://fatafatalina-p7dashboard--streamlit-dashboard-myd8hq.streamlit.app You can test it, chose a client ( scroller on the left hand side) to find out the score for the chosen client number. This makes it easy for a user to understand and have all the information he needs whether it's data, scores and visualizations associated with the prediction.**
---

## Conclusion
This project successfully developed a robust credit scoring model, offering both predictive analysis tools and a dashboard to support decision-making. The **LightGBM** model achieved high performance, and the application of **SMOTE** for class balancing improved prediction quality. Deployment using **Streamlit**, **Flask**, and **Heroku** allows end-users to access predictions and interpret results easily.

