import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')


class ExploratoryDataAnalysis:
    """
    Comprehensive EDA template for regression and classification problems.
    Handles regular, time series, and panel data structures.
    """

    def __init__(self, dataframe, target_column, problem_type='auto',
                 datetime_column=None, panel_id_column=None,
                 sample_threshold=100000, max_categories=50,
                 analysis_mode='standard', categorical_features=None,
                 numeric_features=None, cardinality_threshold=20):
        """
        Initialize EDA analysis.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input dataset
        target_column : str
            Name of target variable
        problem_type : str, default='auto'
            'regression', 'classification', or 'auto' to detect
        datetime_column : str, optional
            Column containing datetime information
        panel_id_column : str, optional
            Column identifying panel/entity IDs
        sample_threshold : int, default=100000
            Row threshold for sampling in visualizations
        max_categories : int, default=50
            Maximum categories to display in categorical plots
        analysis_mode : str, default='standard'
            'quick', 'standard', or 'deep'
        categorical_features : list, optional
            Manually specify which features should be treated as categorical
        numeric_features : list, optional
            Manually specify which features should be treated as numeric
        cardinality_threshold : int, default=20
            For auto-detection: numeric columns with <= this many unique values
            will be treated as categorical
        """

        self.df = dataframe.copy()
        self.target = target_column
        self.datetime_col = datetime_column
        self.panel_id = panel_id_column
        self.sample_threshold = sample_threshold
        self.max_categories = max_categories
        self.analysis_mode = analysis_mode
        self.categorical_features_manual = categorical_features
        self.numeric_features_manual = numeric_features
        self.cardinality_threshold = cardinality_threshold

        # Detect problem type
        if problem_type == 'auto':
            self.problem_type = self._detect_problem_type()
        else:
            self.problem_type = problem_type

        # Detect temporal structure
        self.is_timeseries = self._detect_temporal_structure()
        self.is_panel = self.panel_id is not None

        # Initialize storage
        self.results = {}
        self.figures = {}
        self.warnings = []

    def _detect_problem_type(self):
        """Automatically detect if problem is regression or classification."""
        target_data = self.df[self.target]

        # Check if numeric
        if pd.api.types.is_numeric_dtype(target_data):
            unique_values = target_data.nunique()
            # If fewer than 20 unique values and all integers, likely classification
            if unique_values < 20 and target_data.dropna().apply(float.is_integer).all():
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def _detect_temporal_structure(self):
        """Detect if data has temporal structure."""
        if self.datetime_col is None:
            # Check if any column looks like datetime
            for col in self.df.columns:
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self.datetime_col = col
                    return True
            return False
        else:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.datetime_col]):
                try:
                    self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
                    return True
                except:
                    self.warnings.append(f"Could not convert {self.datetime_col} to datetime")
                    return False
            return True

    def run_full_analysis(self):
        """Run complete EDA pipeline."""
        print("Starting Exploratory Data Analysis...")

        print("1/6 Basic Overview...")
        self.basic_overview()

        print("2/6 Data Quality Check...")
        self.data_quality_check()

        print("3/6 Univariate Analysis...")
        self.univariate_analysis()

        print("4/6 Bivariate Analysis...")
        self.bivariate_analysis()

        if self.is_timeseries:
            print("5/6 Temporal Analysis...")
            self.temporal_analysis()
        else:
            print("5/6 Skipping Temporal Analysis (no time component detected)")

        print("6/6 Multivariate Analysis...")
        self.multivariate_analysis()

        print("\nAnalysis complete! Use generate_report() to view findings.")

    def basic_overview(self):
        """Generate basic dataset overview."""
        overview = {
            'shape': self.df.shape,
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 ** 2,
            'target_column': self.target,
            'problem_type': self.problem_type,
            'is_timeseries': self.is_timeseries,
            'is_panel': self.is_panel
        }

        # Data types summary
        dtype_counts = self.df.dtypes.value_counts()
        overview['dtype_summary'] = dtype_counts.to_dict()

        # Determine column types with smart detection
        all_cols = [col for col in self.df.columns if col != self.target]

        # Start with manual specifications if provided
        if self.categorical_features_manual is not None:
            categorical_cols = [col for col in self.categorical_features_manual if col in all_cols]
        else:
            categorical_cols = []

        if self.numeric_features_manual is not None:
            numeric_cols = [col for col in self.numeric_features_manual if col in all_cols]
        else:
            numeric_cols = []

        # Auto-detect remaining columns
        remaining_cols = [col for col in all_cols if col not in categorical_cols and col not in numeric_cols]
        datetime_cols = []

        for col in remaining_cols:
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_cols.append(col)
            # Check if explicitly object/category type
            elif self.df[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
            # Check if numeric type
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                # Smart detection: if low cardinality, treat as categorical
                n_unique = self.df[col].nunique()
                if n_unique <= self.cardinality_threshold:
                    categorical_cols.append(col)
                    self.warnings.append(
                        f"Column '{col}' treated as categorical (unique values: {n_unique} <= {self.cardinality_threshold})")
                else:
                    numeric_cols.append(col)
            else:
                # Default to categorical for unknown types
                categorical_cols.append(col)

        overview['numeric_columns'] = numeric_cols
        overview['categorical_columns'] = categorical_cols
        overview['datetime_columns'] = datetime_cols
        overview['n_numeric'] = len(numeric_cols)
        overview['n_categorical'] = len(categorical_cols)

        self.results['basic_overview'] = overview

    def data_quality_check(self):
        """Comprehensive data quality assessment."""
        quality = {}

        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        quality['missing_values'] = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        }).sort_values('missing_percentage', ascending=False)

        # Duplicate rows
        quality['n_duplicates'] = self.df.duplicated().sum()
        quality['duplicate_percentage'] = (quality['n_duplicates'] / len(self.df)) * 100

        # Constant/quasi-constant features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        constant_features = []
        for col in numeric_cols:
            if self.df[col].nunique() == 1:
                constant_features.append(col)
        quality['constant_features'] = constant_features

        # High cardinality categoricals
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []
        for col in categorical_cols:
            if self.df[col].nunique() > self.max_categories:
                high_cardinality.append({
                    'column': col,
                    'n_unique': self.df[col].nunique()
                })
        quality['high_cardinality_features'] = pd.DataFrame(high_cardinality)

        # Target-specific checks
        if self.problem_type == 'classification':
            target_dist = self.df[self.target].value_counts()
            quality['class_distribution'] = target_dist
            quality['class_balance_ratio'] = target_dist.min() / target_dist.max()

            if quality['class_balance_ratio'] < 0.1:
                self.warnings.append("Severe class imbalance detected (ratio < 0.1)")

        elif self.problem_type == 'regression':
            quality['target_variance'] = self.df[self.target].var()
            quality['target_std'] = self.df[self.target].std()
            quality['target_range'] = self.df[self.target].max() - self.df[self.target].min()

        # Create missing value heatmap
        if missing.sum() > 0:
            # Sample if needed
            plot_df = self.df if len(self.df) <= self.sample_threshold else \
                self.df.sample(n=self.sample_threshold, random_state=42)

            missing_matrix = plot_df.isnull().astype(int)
            fig = px.imshow(missing_matrix.T,
                            labels=dict(x="Row Index", y="Column", color="Missing"),
                            title="Missing Value Pattern",
                            color_continuous_scale=['lightblue', 'darkred'])
            fig.update_layout(height=max(400, len(self.df.columns) * 15))
            self.figures['missing_heatmap'] = fig

        self.results['data_quality'] = quality

    def univariate_analysis(self):
        """Analyze individual features."""
        univariate = {
            'numeric': {},
            'categorical': {}
        }

        numeric_cols = self.results['basic_overview']['numeric_columns']
        categorical_cols = self.results['basic_overview']['categorical_columns']

        # Numeric features
        for col in numeric_cols:
            col_data = self.df[col].dropna()

            stats_dict = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }

            # Outlier detection (IQR method)
            Q1 = stats_dict['q25']
            Q3 = stats_dict['q75']
            IQR = Q3 - Q1
            outlier_threshold_low = Q1 - 1.5 * IQR
            outlier_threshold_high = Q3 + 1.5 * IQR
            outliers = ((col_data < outlier_threshold_low) | (col_data > outlier_threshold_high)).sum()
            stats_dict['n_outliers'] = outliers
            stats_dict['outlier_percentage'] = (outliers / len(col_data)) * 100

            univariate['numeric'][col] = stats_dict

            # Create distribution plot
            plot_data = col_data if len(col_data) <= self.sample_threshold else \
                col_data.sample(n=self.sample_threshold, random_state=42)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=(f'{col} - Distribution', f'{col} - Box Plot'))

            fig.add_trace(go.Histogram(x=plot_data, name='Histogram', nbinsx=50), row=1, col=1)
            fig.add_trace(go.Box(y=plot_data, name='Box Plot'), row=1, col=2)

            fig.update_layout(title_text=f"Univariate Analysis: {col}", showlegend=False)
            self.figures[f'univariate_numeric_{col}'] = fig

        # Categorical features
        for col in categorical_cols:
            col_data = self.df[col].dropna()
            value_counts = col_data.value_counts()

            stats_dict = {
                'count': len(col_data),
                'n_unique': col_data.nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_frequent_pct': (value_counts.iloc[0] / len(col_data) * 100) if len(value_counts) > 0 else 0
            }

            univariate['categorical'][col] = stats_dict

            # Create bar chart
            if col_data.nunique() > self.max_categories:
                plot_data = value_counts.head(self.max_categories)
                plot_data['Other'] = value_counts.iloc[self.max_categories:].sum()
            else:
                plot_data = value_counts

            fig = px.bar(x=plot_data.index, y=plot_data.values,
                         labels={'x': col, 'y': 'Count'},
                         title=f'Categorical Distribution: {col}')
            fig.update_layout(showlegend=False)
            self.figures[f'univariate_categorical_{col}'] = fig

        self.results['univariate'] = univariate

    def bivariate_analysis(self):
        """Analyze relationships with target variable."""
        bivariate = {
            'numeric': {},
            'categorical': {}
        }

        numeric_cols = self.results['basic_overview']['numeric_columns']
        categorical_cols = self.results['basic_overview']['categorical_columns']

        if self.problem_type == 'regression':
            # Numeric features vs numeric target
            for col in numeric_cols:
                valid_data = self.df[[col, self.target]].dropna()

                # Correlation
                pearson_corr = valid_data[col].corr(valid_data[self.target])
                spearman_corr = valid_data[col].corr(valid_data[self.target], method='spearman')

                bivariate['numeric'][col] = {
                    'pearson_correlation': pearson_corr,
                    'spearman_correlation': spearman_corr
                }

                # Scatter plot
                plot_data = valid_data if len(valid_data) <= self.sample_threshold else \
                    valid_data.sample(n=self.sample_threshold, random_state=42)

                fig = px.scatter(plot_data, x=col, y=self.target,
                                 trendline='ols',
                                 title=f'{col} vs {self.target}<br>Pearson r={pearson_corr:.3f}')
                self.figures[f'bivariate_numeric_{col}'] = fig

            # Categorical features vs numeric target
            for col in categorical_cols:
                if self.df[col].nunique() <= 20:  # Only if reasonable number of categories
                    valid_data = self.df[[col, self.target]].dropna()

                    # ANOVA F-statistic
                    groups = [group[self.target].values for name, group in valid_data.groupby(col)]
                    f_stat, p_value = stats.f_oneway(*groups)

                    bivariate['categorical'][col] = {
                        'f_statistic': f_stat,
                        'p_value': p_value
                    }

                    # Box plot
                    fig = px.box(valid_data, x=col, y=self.target,
                                 title=f'{self.target} by {col}<br>F-stat={f_stat:.3f}, p={p_value:.4f}')
                    self.figures[f'bivariate_categorical_{col}'] = fig

        elif self.problem_type == 'classification':
            # Numeric features vs categorical target
            for col in numeric_cols:
                valid_data = self.df[[col, self.target]].dropna()

                # Sample if needed, stratified by class
                if len(valid_data) > self.sample_threshold:
                    plot_data = valid_data.groupby(self.target, group_keys=False).apply(
                        lambda x: x.sample(n=min(len(x), self.sample_threshold // valid_data[self.target].nunique()),
                                           random_state=42)
                    )
                else:
                    plot_data = valid_data

                # ANOVA or point-biserial
                groups = [group[col].values for name, group in valid_data.groupby(self.target)]
                f_stat, p_value = stats.f_oneway(*groups)

                bivariate['numeric'][col] = {
                    'f_statistic': f_stat,
                    'p_value': p_value
                }

                # Violin plot by class
                fig = px.violin(plot_data, x=self.target, y=col,
                                box=True, points='outliers',
                                title=f'{col} Distribution by {self.target}<br>F-stat={f_stat:.3f}, p={p_value:.4f}')
                self.figures[f'bivariate_numeric_{col}'] = fig

            # Categorical features vs categorical target
            for col in categorical_cols:
                if self.df[col].nunique() <= 20:
                    valid_data = self.df[[col, self.target]].dropna()

                    # Chi-square test
                    contingency = pd.crosstab(valid_data[col], valid_data[self.target])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)

                    bivariate['categorical'][col] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof
                    }

                    # Stacked bar chart
                    fig = px.histogram(valid_data, x=col, color=self.target, barmode='group',
                                       title=f'{col} vs {self.target}<br>Chi2={chi2:.3f}, p={p_value:.4f}')
                    self.figures[f'bivariate_categorical_{col}'] = fig

        self.results['bivariate'] = bivariate

    def temporal_analysis(self):
        """Analyze temporal patterns (if time series data)."""
        if not self.is_timeseries:
            return

        temporal = {}

        # Sort by datetime
        df_sorted = self.df.sort_values(self.datetime_col)

        # Check for gaps
        if len(df_sorted) > 1:
            time_diffs = df_sorted[self.datetime_col].diff()
            temporal['median_time_diff'] = time_diffs.median()
            temporal['time_gaps'] = (time_diffs > 2 * time_diffs.median()).sum()

        # Plot target over time
        fig = px.line(df_sorted, x=self.datetime_col, y=self.target,
                      title=f'{self.target} Over Time')
        self.figures['temporal_target'] = fig

        # For regression, additional time series analysis
        if self.problem_type == 'regression':
            # Note: Advanced decomposition and stationarity tests would require statsmodels
            temporal[
                'note'] = "Advanced temporal analysis (decomposition, ACF/PACF, stationarity tests) requires statsmodels package"

        # Panel data checks
        if self.is_panel:
            balance_check = df_sorted.groupby(self.panel_id)[self.datetime_col].count()
            temporal['is_balanced'] = balance_check.nunique() == 1
            temporal['entities'] = df_sorted[self.panel_id].nunique()
            temporal['obs_per_entity'] = balance_check.describe().to_dict()

        self.results['temporal'] = temporal

    def multivariate_analysis(self):
        """Analyze relationships between features."""
        multivariate = {}

        numeric_cols = self.results['basic_overview']['numeric_columns']

        if len(numeric_cols) > 0:
            # Correlation matrix
            corr_matrix = self.df[numeric_cols + [self.target]].corr()
            multivariate['correlation_matrix'] = corr_matrix

            # Create interactive heatmap
            fig = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            title="Feature Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1)
            fig.update_layout(height=max(500, len(corr_matrix) * 30))
            self.figures['correlation_heatmap'] = fig

            # High correlations (potential multicollinearity)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })

            multivariate['high_correlations'] = pd.DataFrame(high_corr_pairs)

            if len(high_corr_pairs) > 0:
                self.warnings.append(f"Found {len(high_corr_pairs)} feature pairs with |correlation| > 0.8")

        self.results['multivariate'] = multivariate

    def generate_report(self):
        """Generate summary report of findings."""
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Basic overview
        overview = self.results['basic_overview']
        report.append("DATASET OVERVIEW")
        report.append("-" * 80)
        report.append(f"Rows: {overview['n_rows']:,}")
        report.append(f"Columns: {overview['n_columns']}")
        report.append(f"Memory Usage: {overview['memory_usage_mb']:.2f} MB")
        report.append(f"Target Variable: {overview['target_column']}")
        report.append(f"Problem Type: {overview['problem_type']}")
        report.append(f"Numeric Features: {overview['n_numeric']}")
        report.append(f"Categorical Features: {overview['n_categorical']}")
        report.append("")

        # Data quality
        quality = self.results['data_quality']
        report.append("DATA QUALITY")
        report.append("-" * 80)

        missing_summary = quality['missing_values'][quality['missing_values']['missing_count'] > 0]
        if len(missing_summary) > 0:
            report.append(f"Features with missing values: {len(missing_summary)}")
            report.append("Top 5 features by missing percentage:")
            for idx, row in missing_summary.head().iterrows():
                report.append(f"  {idx}: {row['missing_percentage']:.2f}%")
        else:
            report.append("No missing values detected")

        report.append(f"Duplicate rows: {quality['n_duplicates']} ({quality['duplicate_percentage']:.2f}%)")

        if quality['constant_features']:
            report.append(f"Constant features: {', '.join(quality['constant_features'])}")

        if self.problem_type == 'classification':
            report.append(f"Class balance ratio: {quality['class_balance_ratio']:.3f}")

        report.append("")

        # Warnings
        if self.warnings:
            report.append("WARNINGS")
            report.append("-" * 80)
            for warning in self.warnings:
                report.append(f"!! {warning}")
            report.append("")

        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 80)

        if self.problem_type == 'regression' and 'bivariate' in self.results:
            # Top correlated features
            correlations = [(k, v['pearson_correlation']) for k, v in self.results['bivariate']['numeric'].items()]
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            report.append("Top 5 features by correlation with target:")
            for feat, corr in correlations[:5]:
                report.append(f"  {feat}: {corr:.3f}")

        elif self.problem_type == 'classification' and 'bivariate' in self.results:
            # Top discriminative features
            f_stats = [(k, v['f_statistic']) for k, v in self.results['bivariate']['numeric'].items()]
            f_stats.sort(key=lambda x: x[1], reverse=True)

            report.append("Top 5 most discriminative numeric features (by F-statistic):")
            for feat, f_stat in f_stats[:5]:
                report.append(f"  {feat}: {f_stat:.3f}")

        report.append("")
        report.append("=" * 80)
        report.append(f"Total figures generated: {len(self.figures)}")
        report.append("Use .get_figure(name) to view specific plots")
        report.append("Use .export_report(filepath) to save as HTML")
        report.append("=" * 80)

        return "\n".join(report)

    def get_figure(self, figure_name):
        """Retrieve a specific figure."""
        if figure_name in self.figures:
            return self.figures[figure_name]
        else:
            available = list(self.figures.keys())
            print(f"Figure '{figure_name}' not found.")
            print(f"Available figures: {available}")
            return None

    def get_results(self, section_name):
        """Retrieve specific results section."""
        if section_name in self.results:
            return self.results[section_name]
        else:
            available = list(self.results.keys())
            print(f"Section '{section_name}' not found.")
            print(f"Available sections: {available}")
            return None

    def export_report(self, filepath='eda_report.html'):
        """Export complete report with interactive plots to HTML."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        html_content = []
        html_content.append("<html><head><title>EDA Report</title>")
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }")
        html_content.append("h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }")
        html_content.append("h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }")
        html_content.append(
            "pre { background-color: #fff; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }")
        html_content.append(
            ".plot-container { background-color: #fff; padding: 20px; margin: 20px 0; border-radius: 5px; border: 1px solid #ddd; }")
        html_content.append("</style></head><body>")

        html_content.append("<h1>Exploratory Data Analysis Report</h1>")

        # Add text report
        html_content.append("<h2>Summary</h2>")
        html_content.append(f"<pre>{self.generate_report()}</pre>")

        # Add all figures
        html_content.append("<h2>Visualizations</h2>")

        for fig_name, fig in self.figures.items():
            html_content.append(f"<div class='plot-container'>")
            html_content.append(f"<h3>{fig_name.replace('_', ' ').title()}</h3>")
            html_content.append(fig.to_html(include_plotlyjs='cdn', full_html=False))
            html_content.append("</div>")

        html_content.append("</body></html>")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))

        print(f"Report exported to {filepath}")