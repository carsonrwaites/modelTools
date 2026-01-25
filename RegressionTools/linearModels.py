import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


class RegressionAnalysis:
    """
    Comprehensive regression analysis comparing OLS and regularized models.

    LIMITATION! CURRENTLY ONLY HANDLES CONTINUOUS FEATURES
    I still need to add in ability to handle categorical features via dummies

    Additionally: This is only to serve as a starting point for regression analysis.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target : str
        Name of target variable
    features : list, optional
        List of feature column names. If None, uses all columns except target
    drop_features : list, optional
        List of features to exclude from analysis
    test_size : float, default=0.2
        Proportion of data for test set
    random_state : int, default=0
        Random seed for reproducibility
    cv_folds : int, default=5
        Number of cross-validation folds for regularized models
    """

    def __init__(self, data, target, features=None, drop_features=None,
                 test_size=0.2, random_state=0, cv_folds=5):
        self.data = data.copy()
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

        # Determine features
        if features is None:
            self.features = [col for col in data.columns if col != target]
        else:
            self.features = features.copy()

        # Drop specified features
        if drop_features is not None:
            self.features = [f for f in self.features if f not in drop_features]

        # Store results
        self.results = {}
        self.vif_results = None
        self.scaler = None

    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
        return vif_data.sort_values('VIF', ascending=False)

    def prepare_data(self):
        X = self.data[self.features]
        y = self.data[self.target]

        self.vif_results = self.calculate_vif(X)  # Calculate VIF before scaling

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.scaler = StandardScaler()  # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_test_scaled, y_train, y_test

    def calculate_metrics(self, y_true, y_pred, n_features, model_name):
        """
        Calculate regression metrics, fit and error.
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {
            'R2': r2,
            'Adjusted_R2': adj_r2,
            'RMSE': rmse,
            'MAE': mae
        }

    def fit_ols(self, X_train, X_test, y_train, y_test):
        """
        Fit OLS regression using statsmodels.
        """
        # Add constant for statsmodels
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        model = sm.OLS(y_train, X_train_const).fit()

        y_train_pred = model.predict(X_train_const)
        y_test_pred = model.predict(X_test_const)

        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        train_metrics = self.calculate_metrics(
            y_train, y_train_pred, len(self.features), 'OLS'
        )
        test_metrics = self.calculate_metrics(
            y_test, y_test_pred, len(self.features), 'OLS'
        )

        self.results['OLS'] = {
            'model': model,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_residuals': train_residuals,
            'test_residuals': test_residuals,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': model.params,
            'p_values': model.pvalues,
            'cv_score': None  # OLS doesn't have built-in CV
        }

    def fit_ridge(self, X_train, X_test, y_train, y_test):
        """
        Fit Ridge regression with automatic alpha tuning.
        """
        alphas = np.logspace(-3, 3, 100)
        model = RidgeCV(alphas=alphas, cv=self.cv_folds, scoring='neg_mean_squared_error')
        # Note: needed to specify scoring since RidgeCV defaults to R^2.
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        train_metrics = self.calculate_metrics(
            y_train, y_train_pred, len(self.features), 'Ridge'
        )
        test_metrics = self.calculate_metrics(
            y_test, y_test_pred, len(self.features), 'Ridge'
        )

        self.results['Ridge'] = {
            'model': model,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_residuals': train_residuals,
            'test_residuals': test_residuals,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': pd.Series(model.coef_, index=X_train.columns),
            'best_alpha': model.alpha_,
            'cv_score': -model.best_score_,
        }

    def fit_lasso(self, X_train, X_test, y_train, y_test):
        """
        Fit Lasso regression with automatic alpha tuning.
        """
        alphas = np.logspace(-3, 1, 100)
        model = LassoCV(alphas=alphas, cv=self.cv_folds, random_state=self.random_state, max_iter=2000)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        train_metrics = self.calculate_metrics(
            y_train, y_train_pred, len(self.features), 'Lasso'
        )
        test_metrics = self.calculate_metrics(
            y_test, y_test_pred, len(self.features), 'Lasso'
        )

        n_nonzero = np.sum(model.coef_ != 0)

        # Get CV score (mean across folds for best alpha)
        cv_score = np.mean(model.mse_path_[np.where(model.alphas_ == model.alpha_)[0][0]])

        self.results['Lasso'] = {
            'model': model,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_residuals': train_residuals,
            'test_residuals': test_residuals,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': pd.Series(model.coef_, index=X_train.columns),
            'best_alpha': model.alpha_,
            'cv_score': cv_score,
            'n_nonzero_coef': n_nonzero,
        }

    def fit_elasticnet(self, X_train, X_test, y_train, y_test):
        """
        Fit ElasticNet regression with automatic alpha and l1_ratio tuning.
        """
        alphas = np.logspace(-3, 1, 100)
        l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
        model = ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=self.cv_folds,
            random_state=self.random_state,
            max_iter=2000
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        train_metrics = self.calculate_metrics(
            y_train, y_train_pred, len(self.features), 'ElasticNet'
        )
        test_metrics = self.calculate_metrics(
            y_test, y_test_pred, len(self.features), 'ElasticNet'
        )

        # Non-zero coefficients
        n_nonzero = np.sum(model.coef_ != 0)

        # Get CV score (mean across folds for best alpha and l1_ratio)
        # Find the indices of the best parameters
        best_alpha_idx = np.where(model.alphas_ == model.alpha_)[0][0]
        best_l1_idx = np.where(model.l1_ratio == model.l1_ratio_)[0][0]
        cv_score = np.mean(model.mse_path_[best_l1_idx, best_alpha_idx, :])

        self.results['ElasticNet'] = {
            'model': model,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_residuals': train_residuals,
            'test_residuals': test_residuals,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': pd.Series(model.coef_, index=X_train.columns),
            'best_alpha': model.alpha_,
            'best_l1_ratio': model.l1_ratio_,
            'cv_score': cv_score,
            'n_nonzero_coef': n_nonzero,
        }

    def run_analysis(self):
        """
        Run complete regression analysis.
        Future: Do feature selection for OLS? Other reg models?
        """
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data()

        print("Fitting OLS...")
        self.fit_ols(X_train, X_test, y_train, y_test)

        print("Fitting Ridge...")
        self.fit_ridge(X_train, X_test, y_train, y_test)

        print("Fitting Lasso...")
        self.fit_lasso(X_train, X_test, y_train, y_test)

        print("Fitting ElasticNet...")
        self.fit_elasticnet(X_train, X_test, y_train, y_test)

        print("Analysis complete!")

        return self

    def get_summary_table(self):
        """
        Create summary comparison table.
        """
        summary_data = []

        for model_name in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
            result = self.results[model_name]
            row = {
                'Model': model_name,
                'Train_R2': result['train_metrics']['R2'],
                'Test_R2': result['test_metrics']['R2'],
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Test_MAE': result['test_metrics']['MAE'],
            }

            if model_name != 'OLS':
                row['CV_Score'] = result['cv_score']
                # row['Best_Alpha'] = result['best_alpha']  # Not sure if this is helpful

            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def get_coefficients_table(self):
        """
        Create table comparing coefficients across all models.
        Future: Add in OLS p-values here for simplicity?
        """
        coef_dict = {}

        for model_name in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
            result = self.results[model_name]
            coefs = result['coefficients'].copy()

            # For OLS, remove the intercept
            if model_name == 'OLS':
                coefs = coefs.drop('const', errors='ignore')

            coef_dict[model_name] = coefs

        # Combine into DataFrame
        coef_df = pd.DataFrame(coef_dict)
        coef_df.index.name = 'Feature'

        return coef_df

    def plot_performance_comparison(self):
        """
        Create performance comparison bar charts.
        Future: Is this the best visualization for this?"""
        summary = self.get_summary_table()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Comparison', 'RMSE Comparison')
        )

        # R² comparison
        fig.add_trace(
            go.Bar(name='Train R²', x=summary['Model'], y=summary['Train_R2']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Test R²', x=summary['Model'], y=summary['Test_R2']),
            row=1, col=1
        )

        # RMSE comparison
        fig.add_trace(
            go.Bar(name='Train RMSE', x=summary['Model'], y=summary['Train_RMSE']),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Test RMSE', x=summary['Model'], y=summary['Test_RMSE']),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=True, title_text="Model Performance Comparison")
        return fig

    def plot_residuals_comparison(self):
        """
        Plot residuals for all models.
        For the future: compare residuals for test set too?
        """
        # Determine common y-axis limits
        all_residuals = []
        all_fitted = []
        for model_name in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
            all_residuals.extend(self.results[model_name]['test_residuals'])
            all_fitted.extend(self.results[model_name]['test_pred'])

        y_min, y_max = min(all_residuals), max(all_residuals)
        x_min, x_max = min(all_fitted), max(all_fitted)
        y_range = [y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max)]
        x_range = [x_min - 0.1 * abs(x_min), x_max + 0.1 * abs(x_max)]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('OLS', 'Ridge', 'Lasso', 'ElasticNet')
        )

        positions = [(1,1), (1,2), (2,1), (2,2)]

        for (row, col), model_name in zip(positions, ['OLS', 'Ridge', 'Lasso', 'ElasticNet']):
            result = self.results[model_name]

            fig.add_trace(
                go.Scatter(
                    x=result['train_pred'],
                    y=result['train_residuals'],
                    mode='markers',
                    name=model_name,
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=row, col=col)

            # Set common scale
            fig.update_xaxes(range=x_range, title_text="Fitted Values", row=row, col=col)
            fig.update_yaxes(range=y_range, title_text="Residuals", row=row, col=col)

        fig.update_layout(height=600, title_text="Residuals vs Fitted Values")
        return fig

    def plot_qq_comparison(self):
        """
        Plot QQ plots for all models
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('OLS', 'Ridge', 'Lasso', 'ElasticNet')
        )

        positions = [(1,1), (1,2), (2,1), (2,2)]

        for (row, col), model_name in zip(positions, ['OLS', 'Ridge', 'Lasso', 'ElasticNet']):
            result = self.results[model_name]
            residuals = result['train_residuals']

            # Calculate theoretical quantiles
            qq = stats.probplot(residuals, dist="norm")

            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=qq[0][1],
                    mode='markers',
                    name=model_name,
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add reference line
            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=qq[1][1] + qq[1][0] * qq[0][0],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Theoretical Quantiles", row=row, col=col)
            fig.update_yaxes(title_text="Sample Quantiles", row=row, col=col)

        fig.update_layout(height=600, title_text="Q-Q Plots")
        return fig

    def plot_coefficients_comparison(self):
        """
        Compare coefficients across models.
        """
        coef_data = []

        for model_name in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
            result = self.results[model_name]
            coefs = result['coefficients']

            if model_name == 'OLS':
                # Skip intercept for OLS
                coefs = coefs.drop('const', errors='ignore')

            for feature, value in coefs.items():
                coef_data.append({
                    'Feature': feature,
                    'Coefficient': value,
                    'Model': model_name
                })

        df_coef = pd.DataFrame(coef_data)

        fig = px.bar(
            df_coef,
            x='Feature',
            y='Coefficient',
            color='Model',
            barmode='group',
            title='Coefficient Comparison Across Models'
        )

        fig.update_layout(height=500)
        return fig

    def create_dashboard(self):
        """
        Generate complete dashboard with all visualizations.
        Future: Make into HTML file like in the EDA package?
        """
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS DASHBOARD")
        print("="*60)

        # VIF Results
        print("\n--- Variance Inflation Factors ---")
        print(self.vif_results.to_string(index=False))

        # Summary Table
        print("\n--- Model Performance Summary ---")
        print(self.get_summary_table().to_string(index=False))

        # Coefficients Table
        print("\n--- Coefficient Comparison Across Models ---")
        print(self.get_coefficients_table().to_string())

        # OLS-specific details
        print("\n--- OLS Coefficients and P-values ---")
        ols_result = self.results['OLS']
        coef_df = pd.DataFrame({
            'Coefficient': ols_result['coefficients'],
            'P-value': ols_result['p_values']
        })
        print(coef_df.to_string())

        # Regularization parameters
        print("\n--- Regularization Parameters ---")
        for model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            result = self.results[model_name]
            print(f"{model_name}:")
            print(f"  Best Alpha: {result['best_alpha']:.6f}")
            if 'best_l1_ratio' in result:
                print(f"  Best L1 Ratio: {result['best_l1_ratio']:.4f}")
            if 'n_nonzero_coef' in result:
                print(f"  Non-zero Coefficients: {result['n_nonzero_coef']}/{len(self.features)}")

        print("\n" + "="*60)

        # Generate plots
        print("\nGenerating visualizations...")

        plots = {
            'performance_comparison': self.plot_performance_comparison(),
            'residuals_comparison': self.plot_residuals_comparison(),
            'qq_comparison': self.plot_qq_comparison(),
            'coefficients_comparison': self.plot_coefficients_comparison()
        }

        return plots

