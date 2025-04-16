# Housing Price Prediction with Advanced Statistical Analysis
# =========================================================

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function definitions
def load_and_prepare_data(filepath=None):
    """
    Load and prepare housing dataset for analysis.
    If no filepath is provided, generate synthetic data for demonstration.
    """
    if filepath:
        # Load real dataset if provided
        df = pd.read_csv(filepath)
    else:
        # Generate synthetic housing data
        n_samples = 1000
        
        # Create features
        sqft = np.random.normal(2000, 700, n_samples)
        age = np.random.normal(20, 10, n_samples)
        bedrooms = np.random.normal(3, 1, n_samples).round()
        bathrooms = np.random.normal(2, 0.7, n_samples).round(1)
        lot_size = np.random.normal(8000, 2000, n_samples)
        
        # Generate additional features with correlations
        garage = np.random.binomial(1, 0.8, n_samples)
        school_quality = np.random.normal(7, 1.5, n_samples).round()
        school_quality = np.clip(school_quality, 1, 10)
        crime_rate = np.random.exponential(2, n_samples)
        
        # Create neighborhoods
        neighborhoods = ['Downtown', 'Suburb', 'Rural', 'Urban', 'Coastal']
        neighborhood = np.random.choice(neighborhoods, n_samples)
        
        # Generate price with realistic relationships and noise
        price = 50000 + 150 * sqft - 2000 * age + 15000 * bedrooms + 20000 * bathrooms + 0.5 * lot_size
        price += 25000 * garage + 15000 * school_quality - 10000 * crime_rate
        
        # Add neighborhood effects
        neighborhood_effects = {
            'Downtown': 100000,
            'Suburb': 50000,
            'Rural': -50000,
            'Urban': 75000,
            'Coastal': 200000
        }
        
        for i, n in enumerate(neighborhood):
            price[i] += neighborhood_effects[n]
            
        # Add random noise
        price += np.random.normal(0, 50000, n_samples)
        price = np.maximum(price, 10000)  # Ensure no negative prices
        
        # Create the DataFrame
        df = pd.DataFrame({
            'price': price,
            'sqft': sqft,
            'age': age,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'lot_size': lot_size,
            'garage': garage,
            'school_quality': school_quality,
            'crime_rate': crime_rate,
            'neighborhood': neighborhood
        })
        
        # Add some missing values to make the dataset more realistic
        for col in ['sqft', 'age', 'lot_size', 'school_quality']:
            mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
            df.loc[mask, col] = np.nan
    
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values, outliers, and preparing for analysis.
    """
    print("\n=== DATA CLEANING ===")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values by column:")
    print(missing_values[missing_values > 0])
    
    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            # Impute missing values with median for numerical data
            df[col].fillna(df[col].median(), inplace=True)
    
    # Check for and remove outliers using IQR method for key numerical columns
    numerical_cols = ['price', 'sqft', 'age', 'lot_size']
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        print(f"Outliers detected in {col}: {outliers}")
        
        # Instead of removing, we'll cap the outliers for this project
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # Encode categorical variables
    if 'neighborhood' in df.columns:
        df = pd.get_dummies(df, columns=['neighborhood'], drop_first=True)
    
    return df

def perform_eda(df):
    """
    Perform exploratory data analysis on the housing dataset.
    """
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    # Basic statistics
    print("\nBasic statistics for numerical features:")
    print(df.describe().T)
    
    # Distribution of the target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True)
    plt.title('Distribution of Housing Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.savefig('price_distribution.png')
    
    # Check normality of price (for statistical tests)
    plt.figure(figsize=(10, 6))
    stats.probplot(df['price'], plot=plt)
    plt.title('Q-Q Plot of Housing Prices')
    plt.savefig('price_qq_plot.png')
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Feature relationships with price
    important_features = ['sqft', 'age', 'bedrooms', 'bathrooms', 'lot_size', 'school_quality']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(important_features):
        plt.subplot(2, 3, i+1)
        sns.scatterplot(x=feature, y='price', data=df, alpha=0.6)
        plt.title(f'Price vs {feature}')
    plt.tight_layout()
    plt.savefig('price_vs_features.png')
    
    # Calculate and print correlation with target
    print("\nCorrelation with target (price):")
    print(correlation_matrix['price'].sort_values(ascending=False))
    
    return correlation_matrix

def engineer_features(df):
    """
    Engineer new features and select relevant ones for modeling.
    """
    print("\n=== FEATURE ENGINEERING AND SELECTION ===")
    
    # Create new features with safeguards against division by zero and infinite values
    df['price_per_sqft'] = df['price'] / df['sqft'].replace(0, np.nan)
    df['bedroom_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, np.nan)
    df['age_squared'] = df['age'] ** 2
    
    if 'garage' in df.columns and 'school_quality' in df.columns:
        df['garage_school_interaction'] = df['garage'] * df['school_quality']
    
    # Log transformation of price and area (common in real estate analysis)
    df['log_price'] = np.log1p(df['price'])
    df['log_sqft'] = np.log1p(df['sqft'])
    
    # Fill any NaN values that might have been created with median values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Check multicollinearity using VIF
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in features if f not in ['price', 'log_price', 'price_per_sqft']]
    
    X = df[features]
    X = add_constant(X)
    
    # Replace any remaining infinite values with NaN and then fill with median
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVariance Inflation Factor (VIF) for each feature:")
    print(vif_data.sort_values("VIF", ascending=False))
    
    # Filter features with high VIF (multicollinearity)
    high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
    if 'const' in high_vif_features:
        high_vif_features.remove('const')
    
    print(f"\nFeatures with high multicollinearity (VIF > 10): {high_vif_features}")
    
    # For this example, we'll keep all features but in practice you might want to remove some
    # df = df.drop(columns=high_vif_features)
    
    return df

def build_models(df):
    """
    Build and evaluate multiple statistical models.
    """
    print("\n=== STATISTICAL MODELING ===")
    
    # Ensure all numeric columns are float type
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Define target and features
    y = df['price']
    X = df.drop(columns=['price', 'log_price', 'price_per_sqft'])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features for regularized models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store model results
    model_results = {}
    
    # 1. OLS Regression with statsmodels for statistical inference
    print("\n--- OLS Regression with Statistical Inference ---")
    
    # Convert to numpy arrays and ensure float type
    X_train_np = np.asarray(X_train, dtype=float)
    y_train_np = np.asarray(y_train, dtype=float)
    
    # Add constant and ensure float type
    X_train_sm = sm.add_constant(X_train_np)
    X_test_sm = sm.add_constant(np.asarray(X_test, dtype=float))
    
    ols_model = sm.OLS(y_train_np, X_train_sm).fit()
    print(ols_model.summary())
    
    # Make predictions
    ols_predictions = ols_model.predict(X_test_sm)
    ols_rmse = np.sqrt(mean_squared_error(y_test, ols_predictions))
    ols_r2 = r2_score(y_test, ols_predictions)
    
    model_results['OLS Regression'] = {
        'RMSE': ols_rmse,
        'R²': ols_r2,
        'MAE': mean_absolute_error(y_test, ols_predictions)
    }
    
    # 2. Polynomial Regression
    print("\n--- Polynomial Regression ---")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_predictions = poly_model.predict(X_test_poly)
    
    poly_rmse = np.sqrt(mean_squared_error(y_test, poly_predictions))
    poly_r2 = r2_score(y_test, poly_predictions)
    
    model_results['Polynomial Regression'] = {
        'RMSE': poly_rmse,
        'R²': poly_r2,
        'MAE': mean_absolute_error(y_test, poly_predictions)
    }
    
    # 3. Ridge Regression with cross-validation for alpha selection
    print("\n--- Ridge Regression with CV ---")
    alphas = np.logspace(-3, 3, 7)
    
    ridge_cv = GridSearchCV(
        Ridge(), 
        {'alpha': alphas}, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")
    
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_predictions = ridge_model.predict(X_test_scaled)
    
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
    ridge_r2 = r2_score(y_test, ridge_predictions)
    
    model_results['Ridge Regression'] = {
        'RMSE': ridge_rmse,
        'R²': ridge_r2,
        'MAE': mean_absolute_error(y_test, ridge_predictions),
        'Alpha': best_alpha
    }
    
    # 4. Lasso Regression with cross-validation
    print("\n--- Lasso Regression with CV ---")
    lasso_cv = GridSearchCV(
        Lasso(max_iter=10000), 
        {'alpha': alphas}, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    lasso_cv.fit(X_train_scaled, y_train)
    best_alpha_lasso = lasso_cv.best_params_['alpha']
    print(f"Best alpha: {best_alpha_lasso}")
    
    lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    lasso_predictions = lasso_model.predict(X_test_scaled)
    
    # Get feature importance from Lasso
    lasso_coef = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso_model.coef_
    })
    print("\nLasso coefficients (feature importance):")
    print(lasso_coef.sort_values('Coefficient', key=abs, ascending=False).head(10))
    
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_predictions))
    lasso_r2 = r2_score(y_test, lasso_predictions)
    
    model_results['Lasso Regression'] = {
        'RMSE': lasso_rmse,
        'R²': lasso_r2,
        'MAE': mean_absolute_error(y_test, lasso_predictions),
        'Alpha': best_alpha_lasso
    }
    
    # 5. Elastic Net with cross-validation
    print("\n--- Elastic Net with CV ---")
    elastic_cv = GridSearchCV(
        ElasticNet(max_iter=10000),
        {'alpha': alphas, 'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]},
        cv=5,
        scoring='neg_mean_squared_error'
    )
    elastic_cv.fit(X_train_scaled, y_train)
    best_alpha_elastic = elastic_cv.best_params_['alpha']
    best_l1_ratio = elastic_cv.best_params_['l1_ratio']
    print(f"Best parameters: alpha={best_alpha_elastic}, l1_ratio={best_l1_ratio}")
    
    elastic_model = ElasticNet(alpha=best_alpha_elastic, l1_ratio=best_l1_ratio, max_iter=10000)
    elastic_model.fit(X_train_scaled, y_train)
    elastic_predictions = elastic_model.predict(X_test_scaled)
    
    elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_predictions))
    elastic_r2 = r2_score(y_test, elastic_predictions)
    
    model_results['Elastic Net'] = {
        'RMSE': elastic_rmse,
        'R²': elastic_r2,
        'MAE': mean_absolute_error(y_test, elastic_predictions),
        'Alpha': best_alpha_elastic,
        'L1 Ratio': best_l1_ratio
    }
    
    # 6. Random Forest
    print("\n--- Random Forest Regression ---")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    # Feature importance
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })
    print("\nRandom Forest feature importance:")
    print(rf_importance.sort_values('Importance', ascending=False).head(10))
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_r2 = r2_score(y_test, rf_predictions)
    
    model_results['Random Forest'] = {
        'RMSE': rf_rmse,
        'R²': rf_r2,
        'MAE': mean_absolute_error(y_test, rf_predictions)
    }
    
    # 7. Gradient Boosting
    print("\n--- Gradient Boosting Regression ---")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
    gb_r2 = r2_score(y_test, gb_predictions)
    
    model_results['Gradient Boosting'] = {
        'RMSE': gb_rmse,
        'R²': gb_r2,
        'MAE': mean_absolute_error(y_test, gb_predictions)
    }
    
    # 8. Log-transformed OLS model
    print("\n--- Log-transformed OLS Model ---")
    
    # Convert to numpy arrays and ensure float type
    X_train_log = np.asarray(X_train, dtype=float)
    y_train_log = np.log1p(y_train)  # Log transform the target variable
    
    # Add constant and ensure float type
    X_train_log_sm = sm.add_constant(X_train_log)
    X_test_log_sm = sm.add_constant(np.asarray(X_test, dtype=float))
    
    log_model = sm.OLS(y_train_log, X_train_log_sm).fit()
    log_predictions = log_model.predict(X_test_log_sm)
    
    # Handle potential overflow in exp transformation
    max_exp = np.log(np.finfo(np.float64).max) - 1  # Maximum value before overflow
    log_predictions = np.clip(log_predictions, -max_exp, max_exp)
    
    # Transform back to original scale
    log_predictions_transformed = np.expm1(log_predictions)
    
    # Ensure no infinite values
    log_predictions_transformed = np.nan_to_num(log_predictions_transformed, 
                                              nan=np.nanmedian(log_predictions_transformed),
                                              posinf=np.nanmax(log_predictions_transformed),
                                              neginf=np.nanmin(log_predictions_transformed))
    
    log_rmse = np.sqrt(mean_squared_error(y_test, log_predictions_transformed))
    log_r2 = r2_score(y_test, log_predictions_transformed)
    
    model_results['Log-transformed OLS'] = {
        'RMSE': log_rmse,
        'R²': log_r2,
        'MAE': mean_absolute_error(y_test, log_predictions_transformed)
    }
    
    # Compare model performance
    results_df = pd.DataFrame(model_results).T
    print("\nModel Comparison:")
    print(results_df.sort_values('R²', ascending=False))
    
    # Visualize model comparison
    plt.figure(figsize=(12, 8))
    
    # Plot R²
    plt.subplot(2, 1, 1)
    sorted_models = results_df.sort_values('R²', ascending=False).index
    sns.barplot(x=results_df.loc[sorted_models, 'R²'], y=sorted_models, palette='viridis')
    plt.title('Model Comparison - R²')
    plt.xlabel('R² Score (higher is better)')
    
    # Plot RMSE
    plt.subplot(2, 1, 2)
    sns.barplot(x=results_df.loc[sorted_models, 'RMSE'], y=sorted_models, palette='viridis')
    plt.title('Model Comparison - RMSE')
    plt.xlabel('RMSE (lower is better)')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Return the best model name and model object
    best_model_name = results_df.sort_values('R²', ascending=False).index[0]
    best_model = None
    
    if best_model_name == 'OLS Regression':
        best_model = ols_model
    elif best_model_name == 'Polynomial Regression':
        best_model = poly_model
    elif best_model_name == 'Ridge Regression':
        best_model = ridge_model
    elif best_model_name == 'Lasso Regression':
        best_model = lasso_model
    elif best_model_name == 'Elastic Net':
        best_model = elastic_model
    elif best_model_name == 'Random Forest':
        best_model = rf_model
    elif best_model_name == 'Gradient Boosting':
        best_model = gb_model
    elif best_model_name == 'Log-transformed OLS':
        best_model = log_model
    
    return best_model_name, best_model, results_df

def model_diagnostics(df, model_name, model):
    """
    Perform model diagnostics and validation.
    """
    print("\n=== MODEL DIAGNOSTICS AND VALIDATION ===")
    
    # Define X and y based on the best model
    y = df['price']
    X = df.drop(columns=['price', 'log_price', 'price_per_sqft'])
    
    if model_name == 'Log-transformed OLS':
        y = df['log_price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For OLS models, we can perform detailed diagnostics
    if model_name in ['OLS Regression', 'Log-transformed OLS']:
        # Prepare data appropriately
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        
        # Get predictions
        if model_name == 'OLS Regression':
            predictions = model.predict(X_test_sm)
            residuals = y_test - predictions
        else:  # Log-transformed
            log_predictions = model.predict(X_test_sm)
            predictions = np.expm1(log_predictions)
            residuals = y_test - log_predictions  # Residuals in log space
        
        # 1. Residual Analysis
        plt.figure(figsize=(15, 10))
        
        # Residuals vs Fitted values
        plt.subplot(2, 2, 1)
        plt.scatter(model.predict(X_train_sm), model.resid, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals vs Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        
        # Histogram of residuals
        plt.subplot(2, 2, 2)
        sns.histplot(model.resid, kde=True)
        plt.title('Histogram of Residuals')
        
        # Q-Q Plot
        plt.subplot(2, 2, 3)
        stats.probplot(model.resid, plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        # Scale-Location plot
        plt.subplot(2, 2, 4)
        plt.scatter(model.predict(X_train_sm), np.sqrt(np.abs(model.resid)), alpha=0.5)
        plt.title('Scale-Location Plot')
        plt.xlabel('Fitted Values')
        plt.ylabel('√|Residuals|')
        
        plt.tight_layout()
        plt.savefig('residual_diagnostics.png')
        
        # 2. Test for heteroscedasticity
        from statsmodels.stats.diagnostic import het_breuschpagan
        
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        print(f"\nBreusch-Pagan test for heteroscedasticity:")
        print(f"LM statistic: {bp_test[0]:.4f}")
        print(f"p-value: {bp_test[1]:.4f}")
        if bp_test[1] < 0.05:
            print("Conclusion: Reject null hypothesis. Heteroscedasticity present.")
        else:
            print("Conclusion: Fail to reject null hypothesis. No heteroscedasticity detected.")
        
        # 3. Test for normality of residuals
        from scipy.stats import shapiro
        
        stat, p_value = shapiro(model.resid)
        print(f"\nShapiro-Wilk normality test:")
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Conclusion: Reject null hypothesis. Residuals are not normally distributed.")
        else:
            print("Conclusion: Fail to reject null hypothesis. Residuals appear normally distributed.")
    
    # For all models - Cross-validation
    if model_name in ['OLS Regression', 'Log-transformed OLS']:
        # For statsmodels models
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Add constant
            X_fold_train_sm = sm.add_constant(X_fold_train)
            X_fold_val_sm = sm.add_constant(X_fold_val)
            
            # Fit model
            fold_model = sm.OLS(y_fold_train, X_fold_train_sm).fit()
            
            # Predict and calculate R²
            y_pred = fold_model.predict(X_fold_val_sm)
            r2 = r2_score(y_fold_val, y_pred)
            cv_scores.append(r2)
        
        print(f"\nCross-validation results for {model_name}:")
        print(f"Mean R²: {np.mean(cv_scores):.4f}")
        print(f"Std R²: {np.std(cv_scores):.4f}")
        print(f"Individual fold scores: {[round(score, 4) for score in cv_scores]}")
    
    elif model_name in ['Random Forest', 'Gradient Boosting']:
        # For sklearn ensemble models
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-mse_scores)
        
        print(f"\nCross-validation results for {model_name}:")
        print(f"Mean R²: {np.mean(cv_scores):.4f}")
        print(f"Std R²: {np.std(cv_scores):.4f}")
        print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
        print(f"Individual fold R² scores: {[round(score, 4) for score in cv_scores]}")
        
    else:
        # For sklearn linear models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if model_name == 'Polynomial Regression':
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        print(f"\nCross-validation results for {model_name}:")
        print(f"Mean R²: {np.mean(cv_scores):.4f}")
        print(f"Std R²: {np.std(cv_scores):.4f}")
        print(f"Individual fold scores: {[round(score, 4) for score in cv_scores]}")
    
    # Feature importance visualization for the best model
    if model_name in ['OLS Regression', 'Log-transformed OLS']:
        # For regression models

            coefs = pd.DataFrame({
                'Feature': model.model.exog_names[1:],  # Skip the constant
                'Coefficient': model.params[1:],
                'P-value': model.pvalues[1:]
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            plt.figure(figsize=(12, 8))
            colors = ['green' if p < 0.05 else 'red' for p in coefs['P-value']]
            sns.barplot(x='Coefficient', y='Feature', data=coefs.head(15), palette=colors)
            plt.title(f'Top Feature Importance - {model_name}')
            plt.axvline(x=0, color='black', linestyle='-')
            plt.savefig('feature_importance.png')
            
            print("\nTop features by coefficient magnitude:")
            print(coefs.head(10))
            print("\nStatistically significant features (p < 0.05):")
            print(coefs[coefs['P-value'] < 0.05])
            
    elif model_name in ['Random Forest', 'Gradient Boosting']:
        # For tree-based models
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importances.head(15))
        plt.title(f'Feature Importance - {model_name}')
        plt.savefig('feature_importance.png')
        
        print("\nTop features by importance:")
        print(importances.head(10))
        
    elif model_name == 'Lasso Regression' or model_name == 'Elastic Net':
        # For regularized models
        coefs = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coefs.head(15))
        plt.title(f'Feature Coefficients - {model_name}')
        plt.axvline(x=0, color='black', linestyle='-')
        plt.savefig('feature_importance.png')
        
        print("\nTop features by coefficient magnitude:")
        print(coefs.head(10))
        print("\nFeatures with non-zero coefficients:")
        print(coefs[coefs['Coefficient'] != 0])
    
    # Print final conclusion
    print("\nFinal Model Diagnostic Summary:")
    if model_name in ['OLS Regression', 'Log-transformed OLS']:
        print(f"- Adjusted R²: {model.rsquared_adj:.4f}")
        print(f"- AIC: {model.aic:.4f}")
        print(f"- BIC: {model.bic:.4f}")
        print(f"- F-statistic: {model.fvalue:.4f} (p-value: {model.f_pvalue:.4f})")
    
    print(f"- Cross-validation R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    print(f"- Model stability: {'Good' if np.std(cv_scores) < 0.05 else 'Moderate' if np.std(cv_scores) < 0.1 else 'Poor'}")
    
def create_prediction_interface(model_name, model, df):
    """
    Create a simple prediction interface for the best model.
    """
    print("\n=== HOUSING PRICE PREDICTION INTERFACE ===")
    
    # Get feature information for the interface
    X = df.drop(columns=['price', 'log_price', 'price_per_sqft'])
    feature_descriptions = {
        'sqft': 'Square footage of the house',
        'age': 'Age of the house in years',
        'bedrooms': 'Number of bedrooms',
        'bathrooms': 'Number of bathrooms',
        'lot_size': 'Size of the lot in square feet',
        'garage': 'Garage present (1 = Yes, 0 = No)',
        'school_quality': 'School quality rating (1-10)',
        'crime_rate': 'Crime rate in the area',
        'bedroom_ratio': 'Ratio of bedrooms to bathrooms',
        'age_squared': 'Age squared (for non-linear effects)',
        'log_sqft': 'Natural log of square footage',
        'garage_school_interaction': 'Interaction between garage and school quality'
    }
    
    # Add neighborhood columns if they exist
    for col in X.columns:
        if 'neighborhood' in col:
            neighborhood_name = col.replace('neighborhood_', '')
            feature_descriptions[col] = f'Located in {neighborhood_name} (1 = Yes, 0 = No)'
    
    print("\nTo predict house prices, provide the following information:")
    for feature in X.columns:
        if feature in feature_descriptions:
            print(f"- {feature}: {feature_descriptions[feature]}")
        else:
            print(f"- {feature}")
    
    # Example prediction
    print("\nExample prediction:")
    
    # Sample house
    example_house = {
        'sqft': 2000,
        'age': 15,
        'bedrooms': 3,
        'bathrooms': 2,
        'lot_size': 8000,
        'garage': 1,
        'school_quality': 8,
        'crime_rate': 1.5
    }
    
    # Create a DataFrame for the example house
    example_df = pd.DataFrame([example_house])
    
    # Engineer the same features as in the training data
    example_df['bedroom_ratio'] = example_df['bedrooms'] / example_df['bathrooms']
    example_df['age_squared'] = example_df['age'] ** 2
    example_df['log_sqft'] = np.log1p(example_df['sqft'])
    example_df['garage_school_interaction'] = example_df['garage'] * example_df['school_quality']
    
    # Add the neighborhood columns
    for col in X.columns:
        if 'neighborhood' in col and col not in example_df.columns:
            # For simplicity, set all neighborhood values to 0 except one
            if col == 'neighborhood_Suburb':  # Assuming suburb is a common value
                example_df[col] = 1
            else:
                example_df[col] = 0
    
    # Ensure all columns from the training data are present
    for col in X.columns:
        if col not in example_df.columns:
            example_df[col] = 0  # Default value for missing columns
    
    # Make prediction based on the model type
    if model_name in ['OLS Regression', 'Log-transformed OLS']:
        example_df_sm = sm.add_constant(example_df[X.columns])
        prediction = model.predict(example_df_sm)
        if model_name == 'Log-transformed OLS':
            prediction = np.expm1(prediction)  # Transform back from log
    elif model_name == 'Polynomial Regression':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        example_poly = poly.fit_transform(example_df[X.columns])
        prediction = model.predict(example_poly)
    elif model_name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net']:
        scaler = StandardScaler()
        scaler.fit(X)  # Fit on the training data
        example_scaled = scaler.transform(example_df[X.columns])
        prediction = model.predict(example_scaled)
    else:  # Random Forest, Gradient Boosting
        prediction = model.predict(example_df[X.columns])
    
    print(f"\nExample House Details:")
    for feature, value in example_house.items():
        print(f"- {feature}: {value}")
    
    print(f"\nPredicted Price: ${prediction[0]:,.2f}")
    
    print("\n=== PROJECT SUMMARY ===")
    print("This project demonstrates advanced statistical analysis techniques for housing price prediction.")
    print("Key components:")
    print("1. Data cleaning and preprocessing")
    print("2. Exploratory data analysis with statistical visualizations")
    print("3. Feature engineering including transformations and interaction terms")
    print("4. Multiple regression techniques and model comparison")
    print("5. Advanced diagnostics and cross-validation")
    print("6. Practical prediction interface")
    
    print("\nSkills demonstrated:")
    print("- Statistical modeling and inference")
    print("- Handling multicollinearity and outliers")
    print("- Model selection and evaluation")
    print("- Feature importance analysis")
    print("- Regression diagnostics")
    print("- Cross-validation techniques")
    
    return "Project completed successfully"

def main():
    """
    Main function to execute the complete analysis pipeline.
    """
    print("Housing Price Prediction Project")
    print("================================")
    
    # Load and prepare data
    housing_df = load_and_prepare_data()
    
    # Clean the data
    housing_df_clean = clean_data(housing_df)
    
    # Perform EDA
    correlation_matrix = perform_eda(housing_df_clean)
    
    # Engineer features
    housing_df_engineered = engineer_features(housing_df_clean)
    
    # Build and evaluate models
    best_model_name, best_model, model_results = build_models(housing_df_engineered)
    
    # Perform model diagnostics
    model_diagnostics(housing_df_engineered, best_model_name, best_model)
    
    # Create prediction interface
    result = create_prediction_interface(best_model_name, best_model, housing_df_engineered)
    
    print(f"Project completed with best model: {best_model_name}")
    print("See generated visualizations for detailed analysis results.")
    
    return result

if __name__ == "__main__":
    result = main()
    print(result)