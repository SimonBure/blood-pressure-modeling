import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
from datetime import datetime


# ============================================================================
# USER CONFIGURATION
# ============================================================================
chosen_metric = 'mae'  # Options: 'mae', 'rmse', 'mse', 'mape', etc.
family = sm.families.Gamma(link=sm.families.links.Log())  # GLM family
link_function = sm.families.links.Log()  # Link function
alpha = 0.05  # Significance level (Type I error rate)
# ============================================================================


def get_metric_units(metric_name):
    """
    Determine units based on metric name.

    Args:
        metric_name: Name of the error metric

    Returns:
        str: Units for the metric ('mmHg' or 'mmHg²')
    """
    if 'square' in metric_name.lower() or 'sq' in metric_name.lower() or metric_name.lower() in ['mse', 'rmse']:
        return 'mmHg²'
    return 'mmHg'


def save_models_analysis(models_data, output_dir, chosen_metric):
    """
    Save model analysis results to JSON file.

    Args:
        models_data: List of dictionaries containing model results
        output_dir: Directory name (e.g., 'canonical-link', 'log-link')
        chosen_metric: Name of the error metric
    """
    output_path = f"results/stats/{chosen_metric}-models/{output_dir}/performance.json"

    json_data = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models_data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ Saved {len(models_data)} models to {output_path}")


def extract_significant_covariates(results, alpha=0.05):
    """
    Extract significant covariates with their statistics.

    Args:
        results: Fitted GLM results object
        alpha: Significance level (Type I error rate), default 0.05

    Returns:
        dict: {covariate_name: {coefficient, pvalue, ci_lower, ci_upper}}
    """
    significant_covs = {}
    conf_int = results.conf_int(alpha=alpha)

    for param in results.pvalues.index:
        if results.pvalues[param] < alpha and param != 'Intercept':
            significant_covs[param] = {
                'coefficient': float(results.params[param]),
                'pvalue': float(results.pvalues[param]),
                'ci_lower': float(conf_int.loc[param, 0]),
                'ci_upper': float(conf_int.loc[param, 1])
            }

    return significant_covs


def plot_error_distribution(df, chosen_metric, output_path=None):
    """
    Create histogram of error metric distribution for all patients.

    Args:
        df: DataFrame containing error metric
        chosen_metric: Name of the error metric column
        output_path: Path to save the figure (auto-generated if None)
    """
    if output_path is None:
        output_path = f"results/stats/{chosen_metric}-models/hist.png"

    metric_data = df[chosen_metric]
    units = get_metric_units(chosen_metric)

    # Calculate statistics
    mean_val = metric_data.mean()
    median_val = metric_data.median()
    std_val = metric_data.std()

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(metric_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')

    # Add vertical lines for mean and median
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median = {median_val:.2f}')

    # Add statistics text box
    stats_text = f'Mean: {mean_val:.2f} {units}\nMedian: {median_val:.2f} {units}\nStd: {std_val:.2f} {units}\nN: {len(metric_data)}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=11)

    plt.xlabel(f'{chosen_metric.upper()} ({units})', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {chosen_metric.upper()} Across All Patients', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {chosen_metric.upper()} distribution plot saved to {output_path}")


def complete_glm_model_analysis(dataframe, formula, family, output_dir, alpha=0.05):
    """
    Complete GLM model analysis with diagnostics and statistics extraction.

    Args:
        dataframe: Input data
        formula: Model formula string
        family: GLM family object
        output_dir: Directory for saving outputs
        alpha: Significance level (Type I error rate), default 0.05

    Returns:
        dict or None: Model results dict if successful, None if convergence fails
    """
    try:
        model = smf.glm(formula=formula, data=dataframe, family=family)
        results = model.fit()

        # Check convergence
        if not results.converged:
            print(f"⚠ WARNING: Model '{formula}' failed to converge.")
            return None

        # Extract pseudo R-squared (McFadden's)
        pseudo_r2 = 1 - (results.deviance / results.null_deviance)

        # Print summary
        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(results.summary())
        print(f"Pseudo R²: {pseudo_r2:.4f}")
        print(f"{'='*70}\n")

        # Generate diagnostic plots
        # plot_residuals(results, dataframe, output + "_residuals", output_dir)
        # qq_plot(results, output + "_qq", output_dir)

        # Extract significant covariates with full statistics
        significant_covariates = extract_significant_covariates(results, alpha=alpha)

        return {
            'formula': formula,
            'deviance': float(results.deviance),
            'aic': float(results.aic),
            'pseudo_r_squared': float(pseudo_r2),
            'n_observations': int(results.nobs),
            'significant_covariates': significant_covariates
        }

    except Exception as e:
        print(f"⚠ WARNING: Model '{formula}' encountered error: {str(e)}. Skipping...")
        return None


def plot_error_vs_quantitative_covariates(df, quantitative_covariates, chosen_metric, output_path=None):
    """
    Create scatter plots of error metric vs all quantitative covariates.

    Args:
        df: DataFrame containing error metric and covariates
        quantitative_covariates: List of quantitative covariate names
        chosen_metric: Name of the error metric column
        output_path: Path to save the figure (auto-generated if None)
    """
    if output_path is None:
        output_path = f"results/stats/{chosen_metric}-models/{chosen_metric}_vs_quantitative_covariates.png"

    units = get_metric_units(chosen_metric)
    n_covariates = len(quantitative_covariates)
    n_cols = 3
    n_rows = (n_covariates + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_covariates > 1 else [axes]

    for idx, covariate in enumerate(quantitative_covariates):
        ax = axes[idx]
        ax.scatter(df[covariate], df[chosen_metric], alpha=0.6, s=50)

        # Add trend line
        z = np.polyfit(df[covariate], df[chosen_metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[covariate].min(), df[covariate].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        corr = df[covariate].corr(df[chosen_metric])

        ax.set_xlabel(f'{covariate} (standardized)', fontsize=11)
        ax.set_ylabel(f'{chosen_metric.upper()} ({units})', fontsize=11)
        ax.set_title(f'{chosen_metric.upper()} vs {covariate}\n(r = {corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_covariates, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {chosen_metric.upper()} vs quantitative covariates plot saved to {output_path}")


def plot_error_vs_binary_covariates(df, binary_covariates, chosen_metric, output_path=None):
    """
    Create boxplots of error metric vs all binary covariates.

    Args:
        df: DataFrame containing error metric and covariates
        binary_covariates: List of binary covariate names
        chosen_metric: Name of the error metric column
        output_path: Path to save the figure (auto-generated if None)
    """
    if output_path is None:
        output_path = f"results/stats/{chosen_metric}-models/{chosen_metric}_vs_binary_covariates.png"

    units = get_metric_units(chosen_metric)
    n_covariates = len(binary_covariates)
    n_cols = 2
    n_rows = (n_covariates + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten() if n_covariates > 1 else [axes]

    for idx, covariate in enumerate(binary_covariates):
        ax = axes[idx]

        # Separate data by binary value
        data_0 = df[df[covariate] == 0][chosen_metric]
        data_1 = df[df[covariate] == 1][chosen_metric]

        # Create boxplot
        bp = ax.boxplot([data_0, data_1],
                        tick_labels=['0', '1'],
                        patch_artist=True,
                        widths=0.6)

        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points
        ax.scatter([1] * len(data_0), data_0, alpha=0.3, s=30, color='blue')
        ax.scatter([2] * len(data_1), data_1, alpha=0.3, s=30, color='red')

        # Add mean markers
        ax.scatter([1, 2], [data_0.mean(), data_1.mean()],
                  marker='D', s=100, color='green', zorder=3,
                  label='Mean', edgecolors='black', linewidth=1.5)

        # Statistics
        n_0, n_1 = len(data_0), len(data_1)
        mean_0, mean_1 = data_0.mean(), data_1.mean()

        ax.set_ylabel(f'{chosen_metric.upper()} ({units})', fontsize=11)
        ax.set_xlabel(covariate.upper(), fontsize=11)
        ax.set_title(f'{chosen_metric.upper()} vs {covariate.upper()}\n(n={n_0}/{n_1}, μ={mean_0:.2f}/{mean_1:.2f})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)

    # Hide unused subplots
    for idx in range(n_covariates, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {chosen_metric.upper()} vs binary covariates plot saved to {output_path}")


def plot_residuals(results, df, chosen_metric, output="residuals", output_dir="canonical-link"):
    """Plot residuals (kept for potential future use)."""
    residuals = results.resid_deviance
    predictions = results.predict(df)
    units = get_metric_units(chosen_metric)

    plt.figure(figsize=(10, 4))
    plt.scatter(predictions, residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel(f"Predicted {chosen_metric.upper()}")
    plt.ylabel("Deviance Residuals")
    plt.title("Residual Plot")
    plt.savefig(f"results/stats/{chosen_metric}-models/{output_dir}/{output}.png")


def qq_plot(results, chosen_metric, output="qq", output_dir="canonical-link"):
    """Create Q-Q plot (kept for potential future use)."""
    residuals = results.resid_deviance

    plt.figure(figsize=(10, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.savefig(f"results/stats/{chosen_metric}-models/{output_dir}/{output}.png")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
path_quality = f"results/stats/pkpd-quality/opti/{chosen_metric}_vs_covariates.csv"
df = pd.read_csv(path_quality)

df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

quantitative_covariates = ["hr", "pi", "age", "DFG", "E0_indiv", "bmi"]
binary_covariates = ["sex", "HTA", "IECARA", "TABAC"]

scaler = StandardScaler()

df[quantitative_covariates] = scaler.fit_transform(df[quantitative_covariates])

# Transforming binary data from y/n into 1/0 integers.
for bc in binary_covariates:
    if bc != 'sex':  # sex stored as 'f' and 'm'
        df[bc] = [1 if data == "y" else 0 for data in df[bc]]
    else:
        df[bc] = [1 if data == "f" else 0 for data in df[bc]]


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print(f"EXPLORATORY DATA ANALYSIS: {chosen_metric.upper()} vs Covariates")
print("="*70 + "\n")

plot_error_distribution(df, chosen_metric)
plot_error_vs_quantitative_covariates(df, quantitative_covariates, chosen_metric)
plot_error_vs_binary_covariates(df, binary_covariates, chosen_metric)

print("\n" + "="*70)
print("STATISTICAL MODELING")
print("="*70 + "\n")


# ============================================================================
# STATISTICAL MODELING
# ============================================================================
# Define models to test
to_test_model_formulae = [
    f"{chosen_metric} ~ 1",
    f"{chosen_metric} ~ DFG",
    f"{chosen_metric} ~ HTA",
    f"{chosen_metric} ~ IECARA",
    f"{chosen_metric} ~ age",
    f"{chosen_metric} ~ sex",
    f"{chosen_metric} ~ bmi",
    f"{chosen_metric} ~ age + sex + bmi + DFG + HTA + IECARA",
]

# TODO modify the dir to match the link function
# TODO TEST
output_dir = (
    "log-link"
    if type(link_function) is statsmodels.genmod.families.links.Log
    else "canonical-link"
)

# Run all models and collect results
models_results = []

for formula in to_test_model_formulae:
    print(f"\n{'#'*70}")
    print(f"# Processing Model: {formula}")
    print(f"{'#'*70}")

    result = complete_glm_model_analysis(
        dataframe=df,
        formula=formula,
        family=family,
        output_dir=output_dir,
        alpha=alpha
    )

    if result is not None:
        models_results.append(result)

# Save all successful models to JSON
if models_results:
    save_models_analysis(models_results, output_dir, chosen_metric)
    print(f"\n✓ Pipeline complete: {len(models_results)}/{len(to_test_model_formulae)} models saved")
else:
    print("\n⚠ No models converged successfully. No results saved.")
