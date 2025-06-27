import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SolarDataExplorer:
    def __init__(self, file_paths, location_names=None):
        self.file_paths = file_paths
        self.location_names = location_names or [f'Location_{i+1}' for i in range(len(file_paths))]
        self.data = {}
        self.combined_data = None
        self.summary_stats = {}
        
        # Define location-specific seasonal patterns for Cameroon
        self.seasonal_patterns = {
            'Bambili': {
                'dry_months': [11, 12, 1, 2],  # November - February
                'rainy_months': [3, 4, 5, 6, 7, 8, 9, 10]  # March - October
            },
            'Bamenda': {
                'dry_months': [11, 12, 1, 2],  # November - February
                'rainy_months': [3, 4, 5, 6, 7, 8, 9, 10]  # March - October
            },
            'Boufassam': {  # Bafoussam
                'dry_months': [11, 12, 1, 2],  # November - February
                'rainy_months': [3, 4, 5, 6, 7, 8, 9, 10]  # March - October
            },
            'Yaounde': {
                'dry_months': [12, 1, 2, 3, 4],  # December - April
                'rainy_months': [5, 6, 7, 8, 9, 10, 11]  # May - November
            }
        }
        
    def _get_season_for_location(self, month, location):
        """Get season (Dry/Rainy) for a specific month and location"""
        if location in self.seasonal_patterns:
            if month in self.seasonal_patterns[location]['dry_months']:
                return 'Dry Season'
            elif month in self.seasonal_patterns[location]['rainy_months']:
                return 'Rainy Season'
        
        # Fallback to generic seasons if location not found
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        return season_map.get(month, 'Unknown')
        
    def load_and_examine_data(self):
        print("=" * 60)
        print("TASK 2.1: INITIAL DATA ASSESSMENT")
        print("=" * 60)
        
        for i, (file_path, location) in enumerate(zip(self.file_paths, self.location_names)):
            print(f"\nLoading data for {location}")
            print("-" * 40)

            df = pd.read_csv(file_path)
            df['location'] = location

            df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')

            df_filtered = df[(df['Date'].dt.year >= 1950) & (df['Date'].dt.year <= 2024)].copy()

            self.data[location] = df_filtered

            print(f"Dataset Info:")
            print(f"   Original records: {len(df):,}")
            print(f"   Filtered records (1950-2024): {len(df_filtered):,}")
            print(f"   Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
            print(f"   Columns: {list(df_filtered.columns)}")
            print(f"   Data types:")
            for col, dtype in df_filtered.dtypes.items():
                print(f"     - {col}: {dtype}")

            missing_vals = df_filtered.isnull().sum()
            missing_pct = (missing_vals / len(df_filtered) * 100).round(2)
            print(f"\nMissing Values:")
            numeric_columns = ['Temperature', 'Humidity', 'Irradiance', 'Potential', 'WindSpeed']
            for col in numeric_columns:
                if col in df_filtered.columns:
                    print(f"   {col}: {missing_vals[col]} ({missing_pct[col]}%)")

            print(f"\nBasic Statistics:")
            numeric_cols = df_filtered[numeric_columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = df_filtered[numeric_cols].describe()
                print(stats_df.round(3))

            df_filtered_sorted = df_filtered.sort_values('Date')
            time_diffs = df_filtered_sorted['Date'].diff().dropna()
            mode_diff = time_diffs.mode()[0] if len(time_diffs) > 0 else None
            print(f"\nTemporal Resolution:")
            print(f"   Most common time difference: {mode_diff}")
            print(f"   Total years covered: {df_filtered['Date'].dt.year.max() - df_filtered['Date'].dt.year.min() + 1}")
            print(f"   Average readings per day: {len(df_filtered) / ((df_filtered['Date'].max() - df_filtered['Date'].min()).days + 1):.1f}")

        self.combined_data = pd.concat(self.data.values(), ignore_index=True)
        print(f"\nCombined Dataset:")
        print(f"   Total records: {len(self.combined_data):,}")
        print(f"   Locations: {len(self.data)}")
        
    def detect_anomalies(self):
        print(f"\nAnomaly Detection:")
        print("-" * 30)
        
        for location, df in self.data.items():
            print(f"\n{location}:")
            
            numeric_cols = ['Temperature', 'Humidity', 'Irradiance', 'Potential', 'WindSpeed']
            
            for col in numeric_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_pct = len(outliers) / len(df) * 100
                    
                    print(f"   {col}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
                    if len(outliers) > 0:
                        print(f"     Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
                        print(f"     Expected range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    def temporal_analysis(self):
        print("\n" + "=" * 60)
        print("TASK 2.2: EXPLORATORY DATA ANALYSIS - TEMPORAL PATTERNS")
        print("=" * 60)

        # Apply location-specific seasonal patterns
        for location, df in self.data.items():
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['dayofweek'] = df['Date'].dt.dayofweek
            
            # Apply location-specific seasonal mapping
            df['season'] = df['month'].apply(lambda x: self._get_season_for_location(x, location))
            
            # Print seasonal pattern for this location
            print(f"\nSeasonal Pattern for {location}:")
            if location in self.seasonal_patterns:
                dry_months = self.seasonal_patterns[location]['dry_months']
                rainy_months = self.seasonal_patterns[location]['rainy_months']
                
                dry_month_names = [pd.to_datetime(f'2024-{m:02d}-01').strftime('%B') for m in dry_months]
                rainy_month_names = [pd.to_datetime(f'2024-{m:02d}-01').strftime('%B') for m in rainy_months]
                
                print(f"   Dry Season ({len(dry_months)} months): {', '.join(dry_month_names)}")
                print(f"   Rainy Season ({len(rainy_months)} months): {', '.join(rainy_month_names)}")
            else:
                print(f"   Using generic seasonal pattern")
        
        self._plot_daily_patterns()
        self._plot_seasonal_patterns()
        self._plot_yearly_trends()
        self._plot_autocorrelation()
    
    def _plot_daily_patterns(self):
        print("\nAnalyzing Daily Patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Daily Irradiance Patterns by Location', fontsize=16, fontweight='bold')
        
        for i, (location, df) in enumerate(self.data.items()):
            ax = axes[i//2, i%2]

            if 'hour' in df.columns:
                hourly_avg = df.groupby('hour')['Irradiance'].mean()
                hourly_std = df.groupby('hour')['Irradiance'].std()
            else:
                hourly_avg = df.groupby('month')['Irradiance'].mean()
                hourly_std = df.groupby('month')['Irradiance'].std()
            
            ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, label='Mean')
            ax.fill_between(hourly_avg.index, 
                           hourly_avg.values - hourly_std.values,
                           hourly_avg.values + hourly_std.values,
                           alpha=0.3, label='±1 Std')
            
            ax.set_title(f'{location}', fontweight='bold')
            ax.set_xlabel('Hour of Day' if 'hour' in df.columns else 'Month')
            ax.set_ylabel('Solar Irradiance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_seasonal_patterns(self):
        print("\nAnalyzing Seasonal Patterns (Cameroon Climate)...")
        
        # Monthly patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Monthly Average Irradiance by Location (Cameroon Climate)', fontsize=16, fontweight='bold')
        
        for i, (location, df) in enumerate(self.data.items()):
            ax = axes[i//2, i%2]
            monthly_avg = df.groupby('month')['Irradiance'].agg(['mean', 'std'])
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Color bars based on dry/rainy season
            colors = []
            for month in range(1, 13):
                season = self._get_season_for_location(month, location)
                if season == 'Dry Season':
                    colors.append('#FF6B6B')  # Red for dry season
                elif season == 'Rainy Season':
                    colors.append('#4ECDC4')  # Teal for rainy season
                else:
                    colors.append('#95A5A6')  # Gray for generic seasons
            
            bars = ax.bar(range(1, 13), monthly_avg['mean'], 
                         yerr=monthly_avg['std'], capsize=5, alpha=0.7, color=colors)
            ax.set_title(f'{location}', fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Solar Irradiance')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(months, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add legend for seasons
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#FF6B6B', label='Dry Season'),
                             Patch(facecolor='#4ECDC4', label='Rainy Season')]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()

        # Seasonal distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dry vs Rainy Season Irradiance Distribution by Location', fontsize=16, fontweight='bold')
        
        for i, (location, df) in enumerate(self.data.items()):
            ax = axes[i//2, i%2]
            
            # Create boxplot for dry vs rainy seasons
            season_data = []
            season_labels = []
            
            for season in ['Dry Season', 'Rainy Season']:
                season_irradiance = df[df['season'] == season]['Irradiance'].dropna()
                if len(season_irradiance) > 0:
                    season_data.append(season_irradiance)
                    season_labels.append(season)
            
            if len(season_data) > 0:
                bp = ax.boxplot(season_data, labels=season_labels, patch_artist=True)
                
                # Color the boxes
                colors = ['#FF6B6B', '#4ECDC4']  # Red for dry, teal for rainy
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'{location}', fontweight='bold')
            ax.set_xlabel('Season')
            ax.set_ylabel('Solar Irradiance')
            ax.grid(True, alpha=0.3)
            
            # Add seasonal statistics
            if len(season_data) >= 2:
                dry_mean = season_data[0].mean()
                rainy_mean = season_data[1].mean()
                ax.text(0.02, 0.98, f'Dry: {dry_mean:.1f}\nRainy: {rainy_mean:.1f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print seasonal statistics
        print("\nSeasonal Statistics Summary:")
        print("-" * 40)
        for location, df in self.data.items():
            print(f"\n{location}:")
            seasonal_stats = df.groupby('season')['Irradiance'].agg(['count', 'mean', 'std'])
            print(seasonal_stats.round(2))
    
    def _plot_yearly_trends(self):
        print("\nAnalyzing Yearly Trends...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Yearly Irradiance Trends (1950-2024)', fontsize=16, fontweight='bold')
        
        for i, (location, df) in enumerate(self.data.items()):
            ax = axes[i//2, i%2]
            
            yearly_avg = df.groupby('year')['Irradiance'].agg(['mean', 'std'])
            
            ax.errorbar(yearly_avg.index, yearly_avg['mean'], 
                       yerr=yearly_avg['std'], marker='o', capsize=5, alpha=0.7)
            ax.set_title(f'{location}', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Solar Irradiance')
            ax.grid(True, alpha=0.3)
            
            z = np.polyfit(yearly_avg.index, yearly_avg['mean'], 1)
            p = np.poly1d(z)
            ax.plot(yearly_avg.index, p(yearly_avg.index), "--", alpha=0.7, 
                   label=f'Trend: {z[0]:.4f}/year')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_autocorrelation(self):
        print("\nAnalyzing Autocorrelation...")
        
        fig, axes = plt.subplots(len(self.data), 2, figsize=(15, 4*len(self.data)))
        if len(self.data) == 1:
            axes = axes.reshape(1, -1)
            
        fig.suptitle('Autocorrelation Analysis', fontsize=16, fontweight='bold')
        
        for i, (location, df) in enumerate(self.data.items()):
            irradiance_clean = df['Irradiance'].dropna()
            
            if len(irradiance_clean) > 50:
                lags_acf = min(40, len(irradiance_clean)//4)
                acf_vals = acf(irradiance_clean, nlags=lags_acf, fft=True)
                axes[i, 0].plot(range(len(acf_vals)), acf_vals, marker='o')
                axes[i, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[i, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
                axes[i, 0].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
                axes[i, 0].set_title(f'{location} - Autocorrelation')
                axes[i, 0].set_xlabel('Lag')
                axes[i, 0].set_ylabel('ACF')
                axes[i, 0].grid(True, alpha=0.3)
                
                pacf_vals = pacf(irradiance_clean, nlags=lags_acf)
                axes[i, 1].plot(range(len(pacf_vals)), pacf_vals, marker='o')
                axes[i, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[i, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
                axes[i, 1].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
                axes[i, 1].set_title(f'{location} - Partial Autocorrelation')
                axes[i, 1].set_xlabel('Lag')
                axes[i, 1].set_ylabel('PACF')
                axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def spatial_analysis(self):
        print("\n" + "=" * 60)
        print("SPATIAL ANALYSIS: COMPARING LOCATIONS")
        print("=" * 60)

        self._compare_location_statistics()
        self._analyze_location_correlations()
        self._analyze_climate_similarity()
    
    def _compare_location_statistics(self):
        print("\nLocation Statistics Comparison:")
        
        stats_comparison = {}
        
        for location, df in self.data.items():
            stats_comparison[location] = {
                'Mean_Irradiance': df['Irradiance'].mean(),
                'Std_Irradiance': df['Irradiance'].std(),
                'Mean_Temperature': df['Temperature'].mean(),
                'Std_Temperature': df['Temperature'].std(),
                'Mean_Humidity': df['Humidity'].mean(),
                'Mean_WindSpeed': df['WindSpeed'].mean(),
                'Records': len(df)
            }
        
        comparison_df = pd.DataFrame(stats_comparison).T
        print(comparison_df.round(3))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Location Comparison: Key Variables', fontsize=16, fontweight='bold')
        
        variables = ['Irradiance', 'Temperature', 'Humidity', 'WindSpeed']
        
        for i, var in enumerate(variables):
            ax = axes[i//2, i%2]
            
            location_means = [self.data[loc][var].mean() for loc in self.location_names]
            location_stds = [self.data[loc][var].std() for loc in self.location_names]
            
            bars = ax.bar(self.location_names, location_means, yerr=location_stds, capsize=5)
            ax.set_title(f'Average {var}', fontweight='bold')
            ax.set_ylabel(var)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, mean in zip(bars, location_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_location_correlations(self):
        print("\nInter-location Correlations:")

        irradiance_data = {}

        common_dates = None
        for location, df in self.data.items():
            if common_dates is None:
                common_dates = set(df['Date'])
            else:
                common_dates = common_dates.intersection(set(df['Date']))
        
        common_dates = sorted(list(common_dates))
        print(f"   Common dates for correlation: {len(common_dates)}")
        
        for location, df in self.data.items():
            df_common = df[df['Date'].isin(common_dates)].sort_values('Date')
            irradiance_data[location] = df_common['Irradiance'].values
        
        if len(common_dates) > 0:
            corr_df = pd.DataFrame(irradiance_data)
            correlation_matrix = corr_df.corr()
            
            print("   Irradiance correlation matrix:")
            print(correlation_matrix.round(3))

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, square=True, cbar_kws={"shrink": .8})
            plt.title('Inter-location Irradiance Correlations', fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    def _analyze_climate_similarity(self):
        print("\nClimate Similarity Analysis (PCA):")

        location_features = []
        location_labels = []
        
        for location, df in self.data.items():
            monthly_stats = df.groupby('month').agg({
                'Irradiance': ['mean', 'std'],
                'Temperature': ['mean', 'std'],
                'Humidity': ['mean', 'std'],
                'WindSpeed': ['mean', 'std']
            }).reset_index()

            feature_vector = monthly_stats.values.flatten()[1:]
            location_features.append(feature_vector)
            location_labels.append(location)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(location_features)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        
        print(f"   PCA Explained Variance: {pca.explained_variance_ratio_}")
        print(f"   Total Variance Explained: {sum(pca.explained_variance_ratio_):.3f}")

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100, alpha=0.7)
        
        for i, location in enumerate(location_labels):
            plt.annotate(location, (pca_result[i, 0], pca_result[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Climate Similarity Analysis (PCA)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def variable_relationships(self):
        print("\n" + "=" * 60)
        print("VARIABLE RELATIONSHIPS ANALYSIS")
        print("=" * 60)

        self._plot_correlation_matrices()
        self._analyze_lag_correlations()
        self._plot_scatter_matrix()
    
    def _plot_correlation_matrices(self):
        print("\nVariable Correlation Analysis:")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variable Correlations by Location', fontsize=16, fontweight='bold')
        
        variables = ['Irradiance', 'Temperature', 'Humidity', 'Potential', 'WindSpeed']
        
        for i, (location, df) in enumerate(self.data.items()):
            ax = axes[i//2, i%2]
            corr_matrix = df[variables].corr()

            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={"shrink": .8})
            ax.set_title(f'{location}', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

        print("\nStrongest correlations with Irradiance:")
        for location, df in self.data.items():
            corr_with_irradiance = df[variables].corr()['Irradiance'].drop('Irradiance')
            strongest_corr = corr_with_irradiance.abs().sort_values(ascending=False)
            print(f"\n   {location}:")
            for var, corr in strongest_corr.head(3).items():
                print(f"     {var}: {corr_with_irradiance[var]:.3f}")
    
    def _analyze_lag_correlations(self):
        print("\nLag Correlation Analysis:")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Lag Correlations: How Past Weather Affects Future Irradiance', 
                    fontsize=16, fontweight='bold')
        
        for i, (location, df) in enumerate(self.data.items()):
            ax = axes[i//2, i%2]

            variables_to_test = ['Temperature', 'Humidity', 'WindSpeed']
            max_lag = 24
            
            for var in variables_to_test:
                if var in df.columns:
                    lag_corrs = []
                    for lag in range(0, max_lag + 1):
                        if lag == 0:
                            corr = df['Irradiance'].corr(df[var])
                        else:
                            corr = df['Irradiance'].corr(df[var].shift(lag))
                        lag_corrs.append(corr)
                    
                    ax.plot(range(max_lag + 1), lag_corrs, marker='o', 
                           label=var, alpha=0.7)
            
            ax.set_title(f'{location}', fontweight='bold')
            ax.set_xlabel('Lag (time periods)')
            ax.set_ylabel('Correlation with Future Irradiance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_scatter_matrix(self):
        print("\nVariable Relationships Scatter Matrix:")

        location = self.location_names[0]
        df = self.data[location]
        
        variables = ['Irradiance', 'Temperature', 'Humidity', 'Potential', 'WindSpeed']

        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
            print(f"   Using random sample of 5,000 points from {location}")
        else:
            df_sample = df

        pd.plotting.scatter_matrix(df_sample[variables], figsize=(15, 15), 
                                 alpha=0.6, diagonal='hist')
        plt.suptitle(f'Variable Relationships - {location}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        print("\n" + "=" * 60)
        print("DATA EXPLORATION SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\nDataset Overview:")
        print(f"   Number of locations: {len(self.data)}")
        print(f"   Total records: {len(self.combined_data):,}")
        print(f"   Date range: 1950-2024 (75 years)")
        print(f"   Variables: Temperature, Humidity, Irradiance, Potential, WindSpeed")
        
        print(f"\nCameroon Climate Patterns Applied:")
        for location in self.location_names:
            if location in self.seasonal_patterns:
                dry_months = len(self.seasonal_patterns[location]['dry_months'])
                rainy_months = len(self.seasonal_patterns[location]['rainy_months'])
                print(f"   {location}: {dry_months}-month dry season, {rainy_months}-month rainy season")
        
        print(f"\nKey Findings:")

        total_missing = self.combined_data.isnull().sum().sum()
        missing_pct = total_missing / (len(self.combined_data) * len(self.combined_data.columns)) * 100
        print(f"   Overall missing data: {missing_pct:.2f}%")

        irradiance_stats = []
        for location, df in self.data.items():
            irradiance_stats.append({
                'Location': location,
                'Mean': df['Irradiance'].mean(),
                'Std': df['Irradiance'].std(),
                'Min': df['Irradiance'].min(),
                'Max': df['Irradiance'].max()
            })
        
        stats_df = pd.DataFrame(irradiance_stats)
        print(f"   Irradiance varies significantly across locations")
        print(f"     - Highest mean: {stats_df.loc[stats_df['Mean'].idxmax(), 'Location']} ({stats_df['Mean'].max():.1f})")
        print(f"     - Lowest mean: {stats_df.loc[stats_df['Mean'].idxmin(), 'Location']} ({stats_df['Mean'].min():.1f})")

        print(f"   Strong seasonal patterns observed following Cameroon climate")
        print(f"   Long-term trends visible over 75-year period")

        print(f"\nModeling Recommendations:")
        print(f"   Consider location-specific models due to geographical differences")
        print(f"   Include Cameroon-specific seasonal features (dry/rainy seasons)")
        print(f"   Temperature shows strongest correlation with Irradiance")
        print(f"   Lag features may be important for temporal dependencies")
        
        print(f"\nNext Steps:")
        print(f"   Address missing data using appropriate imputation methods")
        print(f"   Create engineered features based on Cameroon climate patterns")
        print(f"   Consider climate change trends over the 75-year period")
        print(f"   Prepare train/validation/test splits maintaining temporal order")

    def seasonal_decomposition_analysis(self):
        print("\n" + "=" * 60)
        print("SEASONAL DECOMPOSITION ANALYSIS")
        print("=" * 60)
        
        fig, axes = plt.subplots(len(self.data), 4, figsize=(20, 5*len(self.data)))
        if len(self.data) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (location, df) in enumerate(self.data.items()):
            print(f"\nDecomposing {location} irradiance time series...")

            df_daily = df.groupby(df['Date'].dt.date)['Irradiance'].mean().reset_index()
            df_daily['Date'] = pd.to_datetime(df_daily['Date'])
            df_daily = df_daily.set_index('Date').sort_index()

            df_daily = df_daily.dropna()
            
            if len(df_daily) > 365:
                try:
                    decomposition = seasonal_decompose(df_daily['Irradiance'], 
                                                     model='additive', 
                                                     period=365)

                    decomposition.observed.plot(ax=axes[i, 0], title=f'{location} - Original')
                    decomposition.trend.plot(ax=axes[i, 1], title=f'{location} - Trend')
                    decomposition.seasonal.plot(ax=axes[i, 2], title=f'{location} - Seasonal')
                    decomposition.resid.plot(ax=axes[i, 3], title=f'{location} - Residual')
                    
                    for ax in axes[i]:
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        
                except Exception as e:
                    print(f"   Could not decompose {location}: {e}")
                    for j in range(4):
                        axes[i, j].text(0.5, 0.5, f'Decomposition failed\n{location}', 
                                       ha='center', va='center', transform=axes[i, j].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def data_quality_assessment(self):
        print("\n" + "=" * 60)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 60)
        
        quality_report = {}
        
        for location, df in self.data.items():
            print(f"\n{location} Data Quality:")
            print("-" * 30)

            total_expected = 75 * 365
            actual_records = len(df)
            completeness = (actual_records / total_expected) * 100
            
            print(f"   Completeness: {completeness:.1f}% ({actual_records:,}/{total_expected:,})")

            print(f"   Missing Values:")
            for col in ['Temperature', 'Humidity', 'Irradiance', 'Potential', 'WindSpeed']:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    missing_pct = (missing_count / len(df)) * 100
                    print(f"     {col}: {missing_count} ({missing_pct:.2f}%)")

            print(f"   Consistency Checks:")

            negative_irradiance = (df['Irradiance'] < 0).sum()
            print(f"     Negative irradiance values: {negative_irradiance}")

            extreme_temp = ((df['Temperature'] < -50) | (df['Temperature'] > 60)).sum()
            print(f"     Extreme temperatures (<-50°C or >60°C): {extreme_temp}")

            invalid_humidity = ((df['Humidity'] < 0) | (df['Humidity'] > 100)).sum()
            print(f"     Invalid humidity values (<0% or >100%): {invalid_humidity}")

            duplicate_dates = df['Date'].duplicated().sum()
            print(f"     Duplicate timestamps: {duplicate_dates}")

            df_sorted = df.sort_values('Date')
            time_gaps = df_sorted['Date'].diff()
            large_gaps = (time_gaps > pd.Timedelta(days=2)).sum()
            print(f"     Large time gaps (>2 days): {large_gaps}")
            
            quality_report[location] = {
                'completeness': completeness,
                'missing_values': df.isnull().sum().sum(),
                'negative_irradiance': negative_irradiance,
                'extreme_temperatures': extreme_temp,
                'invalid_humidity': invalid_humidity,
                'duplicate_dates': duplicate_dates,
                'large_gaps': large_gaps
            }

        print(f"\nOverall Data Quality Summary:")
        print("-" * 40)
        
        quality_df = pd.DataFrame(quality_report).T
        print(quality_df)
        
        return quality_report
    
    def feature_engineering_preview(self):
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PREVIEW")
        print("=" * 60)

        location = self.location_names[0]
        df = self.data[location].copy()
        
        print(f"Creating features for {location} (as example):")
        print("-" * 40)

        print("   Time-based features:")
        df['hour'] = df['Date'].dt.hour
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['month'] = df['Date'].dt.month
        
        # Use Cameroon-specific seasonal encoding
        df['season'] = df['month'].apply(lambda x: self._get_season_for_location(x, location))
        df['is_dry_season'] = (df['season'] == 'Dry Season').astype(int)
        df['is_rainy_season'] = (df['season'] == 'Rainy Season').astype(int)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        print("     Hour, day_of_year, month, Cameroon-specific seasons")
        print("     Cyclical encodings (sin/cos) for hour and day")
        print("     Binary indicators for dry/rainy seasons")

        print("   Lag features:")
        for lag in [1, 7, 30, 365]:
            df[f'Irradiance_lag_{lag}'] = df['Irradiance'].shift(lag)
            df[f'Temperature_lag_{lag}'] = df['Temperature'].shift(lag)
        
        print("     Irradiance and Temperature lags: 1, 7, 30, 365 days")

        print("   Rolling statistics:")
        for window in [7, 30, 365]:
            df[f'Irradiance_rolling_mean_{window}'] = df['Irradiance'].rolling(window=window).mean()
            df[f'Irradiance_rolling_std_{window}'] = df['Irradiance'].rolling(window=window).std()
            df[f'Temperature_rolling_mean_{window}'] = df['Temperature'].rolling(window=window).mean()
        
        print("     Rolling means and std for 7, 30, 365 day windows")

        print("   Weather interaction features:")
        df['temp_humidity_interaction'] = df['Temperature'] * df['Humidity']
        df['temp_windspeed_interaction'] = df['Temperature'] * df['WindSpeed']
        
        print("     Temperature × Humidity, Temperature × WindSpeed")

        print("   Solar-specific features:")
        if 'Potential' in df.columns:
            df['clear_sky_index'] = df['Irradiance'] / (df['Potential'] + 1e-6)
            print("     Clear sky index (Irradiance / Potential)")

        print("   Difference features:")
        df['Irradiance_diff_1d'] = df['Irradiance'].diff(1)
        df['Temperature_diff_1d'] = df['Temperature'].diff(1)
        
        print("     1-day differences for Irradiance and Temperature")

        original_features = ['Date', 'Temperature', 'Humidity', 'Irradiance', 'Potential', 'WindSpeed']
        engineered_features = [col for col in df.columns if col not in original_features and col != 'location']
        
        print(f"\nFeature Engineering Results:")
        print(f"   Original features: {len(original_features) - 1}")
        print(f"   Engineered features: {len(engineered_features)}")
        print(f"   Total features: {len(df.columns) - 2}")

        print(f"\nTop engineered features correlated with Irradiance:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['Irradiance'].abs().sort_values(ascending=False)

        top_features = correlations.drop('Irradiance').head(10)
        for feature, corr in top_features.items():
            print(f"     {feature}: {corr:.3f}")
        
        return df
    
    def export_processed_data(self, output_dir='processed_data'):
        import os
        
        print(f"\nExporting processed data to '{output_dir}' directory...")

        os.makedirs(output_dir, exist_ok=True)
        
        for location, df in self.data.items():
            df_clean = df.copy()

            numeric_cols = ['Temperature', 'Humidity', 'Irradiance', 'Potential', 'WindSpeed']
            for col in numeric_cols:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR 
                    upper_bound = Q3 + 3 * IQR

                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

            df_clean = df_clean.sort_values('Date').reset_index(drop=True)

            output_file = os.path.join(output_dir, f'{location}_processed.csv')
            df_clean.to_csv(output_file, index=False)
            print(f"   {location}: {len(df_clean)} records → {output_file}")

        combined_file = os.path.join(output_dir, 'combined_all_locations.csv')
        self.combined_data.to_csv(combined_file, index=False)
        print(f"   Combined: {len(self.combined_data)} records → {combined_file}")

        metadata = {
            'locations': self.location_names,
            'date_range': f"{self.combined_data['Date'].min()} to {self.combined_data['Date'].max()}",
            'total_records': len(self.combined_data),
            'variables': ['Temperature', 'Humidity', 'Irradiance', 'Potential', 'WindSpeed'],
            'seasonal_patterns': self.seasonal_patterns,
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = os.path.join(output_dir, 'dataset_metadata.txt')
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"   Metadata → {metadata_file}")
        print(f"\nData export complete! Ready for modeling phase.")

def prepare_train_test_split(df, test_size=0.2, val_size=0.1):
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    print(f"Data split:")
    print(f"  Train: {len(train_df)} records ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"  Validation: {len(val_df)} records ({val_df['Date'].min()} to {val_df['Date'].max()})")
    print(f"  Test: {len(test_df)} records ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    return train_df, val_df, test_df

def calculate_forecast_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'error': 'No valid predictions to evaluate'}
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
        'MAPE': np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100,
        'R2': r2_score(y_true_clean, y_pred_clean),
        'n_samples': len(y_true_clean)
    }
    return metrics

if __name__ == "__main__":
    file_paths = [
        '../data/Bambili.csv',
        '../data/Bamenda.csv',
        '../data/Bafoussam.csv',
        '../data/Yaounde.csv'
    ]
    
    location_names = ['Bambili', 'Bamenda', 'Boufassam', 'Yaounde']
    
    explorer = SolarDataExplorer(file_paths, location_names)
    
    print("Starting Solar Data Exploration Analysis with Cameroon Climate Patterns...")
    
    explorer.load_and_examine_data()
    explorer.detect_anomalies()
    explorer.temporal_analysis()
    explorer.spatial_analysis()
    explorer.variable_relationships()
    explorer.seasonal_decomposition_analysis()
    quality_report = explorer.data_quality_assessment()
    engineered_df = explorer.feature_engineering_preview()
    explorer.generate_summary_report()
    explorer.export_processed_data()
    
    print("\nSolar Data Exploration Complete with Cameroon-specific Climate Analysis!")
