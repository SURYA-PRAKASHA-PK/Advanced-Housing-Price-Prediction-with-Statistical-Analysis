# Advanced Housing Price Prediction with Statistical Analysis

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview
This project implements a comprehensive housing price prediction system using advanced statistical analysis techniques. The system incorporates multiple regression models, feature engineering, and thorough model diagnostics to provide accurate price predictions and insights into housing market dynamics.

## Key Features
- Advanced statistical modeling and inference
- Multicollinearity analysis using Variance Inflation Factor (VIF)
- Multiple regression techniques (OLS, Ridge, Lasso, Elastic Net)
- Machine learning models (Random Forest, Gradient Boosting)
- Log-transformed modeling for non-linear relationships
- Comprehensive model diagnostics and validation
- Interactive prediction interface

## Technical Implementation

### Data Processing Pipeline
1. **Data Loading and Preparation**
   - Synthetic data generation for demonstration
   - Real-world data compatibility
   - Missing value handling
   - Outlier detection and treatment

2. **Feature Engineering**
   - Price per square foot calculation
   - Bedroom-bathroom ratio
   - Age squared for non-linear effects
   - Log transformations
   - Interaction terms
   - Categorical variable encoding

3. **Statistical Analysis**
   - Correlation analysis
   - Multicollinearity detection
   - Normality testing
   - Heteroscedasticity analysis

### Model Architecture

#### 1. OLS Regression
- Statistical inference capabilities
- P-value analysis
- Confidence intervals
- Residual analysis

#### 2. Regularized Models
- Ridge Regression with cross-validation
- Lasso Regression for feature selection
- Elastic Net for balanced regularization

#### 3. Machine Learning Models
- Random Forest with feature importance
- Gradient Boosting with hyperparameter tuning
- Cross-validation for model selection

#### 4. Log-transformed Models
- Handling non-linear relationships
- Proper back-transformation
- Overflow prevention
- Infinite value handling

### Model Diagnostics
- Residual analysis
- Q-Q plots
- Scale-location plots
- Breusch-Pagan test for heteroscedasticity
- Shapiro-Wilk test for normality
- Cross-validation performance metrics

## Technical Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn
- SciPy

## Project Structure
```
house-statistics/
├── main.py              # Main implementation
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
└── data/               # Data directory
```

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python main.py
```

## Model Selection Criteria
- R² score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Cross-validation performance
- Statistical significance
- Model interpretability

## Advanced Features

### 1. Multicollinearity Handling
- VIF calculation and analysis
- Feature selection based on VIF thresholds
- Correlation matrix visualization

### 2. Model Validation
- K-fold cross-validation
- Train-test split
- Performance metrics comparison
- Residual diagnostics

### 3. Feature Importance
- Coefficient analysis
- Statistical significance
- Variable contribution
- Interaction effects

### 4. Prediction Interface
- User-friendly input handling
- Feature validation
- Prediction confidence intervals
- Model explanation

## Statistical Methodology

### 1. Data Preprocessing
- Missing value imputation using median
- Outlier treatment using IQR method
- Feature scaling and normalization
- Categorical variable encoding

### 2. Model Building
- Feature selection using statistical tests
- Model specification
- Parameter estimation
- Hypothesis testing

### 3. Model Evaluation
- Residual analysis
- Goodness-of-fit tests
- Prediction accuracy
- Model stability

### 4. Advanced Diagnostics
- Heteroscedasticity testing
- Normality testing
- Multicollinearity detection
- Model comparison

## Performance Metrics
- R² (Coefficient of Determination)
- Adjusted R²
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)

## Future Enhancements
1. Deep learning integration
2. Time series analysis
3. Geospatial analysis
4. Automated feature engineering
5. Real-time prediction API
6. Advanced visualization dashboard

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Statsmodels development team
- Scikit-learn contributors
- Scientific Python community 
