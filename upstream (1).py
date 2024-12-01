import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV

# Function to plot Taylor Diagram
def plot_taylor_diagram(y_true, y_pred, title):
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    std_obs = np.std(y_true)
    std_model = np.std(y_pred)

    plt.figure(figsize=(8, 8))
    plt.scatter(std_obs, corr_coef, color="b", marker="o", label="Model")
    plt.plot([0, std_obs], [corr_coef, corr_coef], "b--", label="Correlation")
    plt.plot([std_obs, std_obs], [0, corr_coef], "b--")
    plt.plot([0, std_obs], [0, corr_coef], "b--")

    plt.scatter(std_model, corr_coef, color="r", marker="o", label="Observations")
    plt.plot([0, std_model], [corr_coef, corr_coef], "r--", label="Correlation")
    plt.plot([std_model, std_model], [0, corr_coef], "r--")
    plt.plot([0, std_model], [0, corr_coef], "r--")

    plt.xlabel("Standard Deviation of Observations")
    plt.ylabel("Correlation Coefficient")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app
st.title("Random Forest Regression with Feature Selection and Hyperparameter Tuning")

# Step 1: Load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Step 2: Data Preprocessing
    # Handle missing values
    data = data.dropna()

    # Ask user for target column name
    target_column = st.text_input("Enter the target column name:", "MMP (mPa)")

    if target_column in data.columns:
        # Ensure target variable is numeric
        y = pd.to_numeric(data[target_column], errors='coerce')

        # Handle any categorical variables in X
        X = pd.get_dummies(data.drop(target_column, axis=1), drop_first=True)

        # Step 3: Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection using RFE with Cross-validation
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rfecv = RFECV(estimator=model_rf, step=1, cv=5, scoring='neg_mean_squared_error')
        rfecv.fit(X_scaled, y)

        # Get the selected features and their ranking
        selected_features = X.columns[rfecv.support_].tolist()
        st.write("Selected Features:")
        st.write(selected_features)

        # Step 4: Hyperparameter tuning with RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        model_rf_tuned = RandomForestRegressor(random_state=42)
        rf_random = RandomizedSearchCV(estimator=model_rf_tuned,
                                       param_distributions=param_dist,
                                       n_iter=50,
                                       cv=5,
                                       scoring="neg_mean_squared_error",
                                       verbose=1,
                                       random_state=42)
        rf_random.fit(X_scaled, y)

        st.write("Best Hyperparameters:")
        st.write(rf_random.best_params_)

        # Step 5: Model Evaluation on Testing Set
        X_selected = rfecv.transform(X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X_selected,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        model_rf_tuned_best = RandomForestRegressor(**rf_random.best_params_,
                                                    random_state=42)
        model_rf_tuned_best.fit(X_train, y_train)
        y_test_pred = model_rf_tuned_best.predict(X_test)

        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        st.write(f"Mean Squared Error (MSE) on Test Set: {mse_test}")
        st.write(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")
        st.write(f"R-squared (R2) Score on Test Set: {r2_test}")

        # Step 6: Visualization

        # Feature Importances Plot
        feature_importances = model_rf_tuned_best.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=selected_features,
                    y=feature_importances,
                    palette='viridis')
        plt.xlabel("Selected Features")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        st.pyplot(plt)

        # Residual Plot
        residuals = y_test - y_test_pred
        plt.scatter(y_test,
                    residuals,
                    alpha=0.7,
                    color='b')
        plt.axhline(y=0,
                    color='k',
                    linestyle='--')
        plt.xlabel("Actual MPG (mpg)")
        plt.ylabel("Residuals (mpg)")
        plt.title("Residual Plot")
        plt.tight_layout()
        st.pyplot(plt)

        # Scatter Plot of Predicted vs. Actual Values with Confidence Interval
        plt.scatter(y_test,
                    y_test_pred,
                    alpha=0.7,
                    color='b',
                    edgecolors='k')
        plt.plot([min(y_test), max(y_test)],
                 [min(y_test), max(y_test)],
                 linestyle="--",
                 linewidth=2)
        plt.xlabel("Actual MPG (mpg)")
        plt.ylabel("Predicted MPG (mpg)")
        plt.title("Scatter Plot of Predicted vs. Actual Values")
        plt.tight_layout()
        st.pyplot(plt)

        # Step 7: Taylor Diagram
        plot_taylor_diagram(y_test,
                            y_test_pred,
                            title="Taylor Diagram")
    else:
        st.error("Target column not found in the dataset. Please check the column name.")
else:
    st.info("Please upload a CSV file to proceed.")