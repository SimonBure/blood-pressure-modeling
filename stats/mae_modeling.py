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


def save_models_analysis(models_data, output_dir):
    """
    Save model analysis results to JSON file.

    Args:
        models_data: List of dictionaries containing model results
        output_dir: Directory name (e.g., 'canonical-link', 'log-link')
    """
    output_path = f"results/stats/mae-models/{output_dir}/performance.json"

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


def complete_glm_model_analysis(dataframe, formula, family, output, output_dir, alpha=0.05):
    """
    Complete GLM model analysis with diagnostics and statistics extraction.

    Args:
        dataframe: Input data
        formula: Model formula string
        family: GLM family object
        output: Output name for plots and results
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
            print(f"⚠ WARNING: Model '{output}' failed to converge.")
            return None

        # Extract pseudo R-squared (McFadden's)
        pseudo_r2 = 1 - (results.deviance / results.null_deviance)

        # Print summary
        print(f"\n{'='*70}")
        print(f"Model: {output}")
        print(f"Formula: {formula}")
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
            'name': output,
            'formula': formula,
            'deviance': float(results.deviance),
            'aic': float(results.aic),
            'pseudo_r_squared': float(pseudo_r2),
            'n_observations': int(results.nobs),
            'significant_covariates': significant_covariates
        }

    except Exception as e:
        print(f"⚠ WARNING: Model '{output}' encountered error: {str(e)}. Skipping...")
        return None


def plot_mae_vs_quantitative_covariates(df, quantitative_covariates, output_path="results/stats/mae-models/mae_vs_quantitative_covariates.png"):
    """
    Create scatter plots of MAE vs all quantitative covariates.

    Args:
        df: DataFrame containing MAE and covariates
        quantitative_covariates: List of quantitative covariate names
        output_path: Path to save the figure
    """
    n_covariates = len(quantitative_covariates)
    n_cols = 3
    n_rows = (n_covariates + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_covariates > 1 else [axes]

    for idx, covariate in enumerate(quantitative_covariates):
        ax = axes[idx]
        ax.scatter(df[covariate], df['mae'], alpha=0.6, s=50)

        # Add trend line
        z = np.polyfit(df[covariate], df['mae'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[covariate].min(), df[covariate].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        corr = df[covariate].corr(df['mae'])

        ax.set_xlabel(f'{covariate} (standardized)', fontsize=11)
        ax.set_ylabel('MAE (mmHg)', fontsize=11)
        ax.set_title(f'MAE vs {covariate}\n(r = {corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_covariates, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ MAE vs quantitative covariates plot saved to {output_path}")


def plot_mae_vs_binary_covariates(df, binary_covariates, output_path="results/stats/mae-models/mae_vs_binary_covariates.png"):
    """
    Create boxplots of MAE vs all binary covariates.

    Args:
        df: DataFrame containing MAE and covariates
        binary_covariates: List of binary covariate names
        output_path: Path to save the figure
    """
    n_covariates = len(binary_covariates)
    n_cols = 2
    n_rows = (n_covariates + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten() if n_covariates > 1 else [axes]

    for idx, covariate in enumerate(binary_covariates):
        ax = axes[idx]

        # Separate data by binary value
        data_0 = df[df[covariate] == 0]['mae']
        data_1 = df[df[covariate] == 1]['mae']

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

        ax.set_ylabel('MAE (mmHg)', fontsize=11)
        ax.set_xlabel(covariate.upper(), fontsize=11)
        ax.set_title(f'MAE vs {covariate.upper()}\n(n={n_0}/{n_1}, μ={mean_0:.2f}/{mean_1:.2f})',
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
    print(f"✓ MAE vs binary covariates plot saved to {output_path}")


def plot_residuals(results, df, output="residuals", output_dir="canonical-link"):
    residuals = results.resid_deviance

    predictions = results.predict(df)

    plt.figure(figsize=(10, 4))
    plt.scatter(predictions, residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted MAE")
    plt.ylabel("Deviance Residuals")
    plt.title("Residual Plot")
    plt.savefig(f"results/stats/mae-models/{output_dir}/{output}.png")


def qq_plot(results, output="qq", output_dir="canonical-link"):
    residuals = results.resid_deviance

    plt.figure(figsize=(10, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.savefig(f"results/stats/mae-models/{output_dir}/{output}.png")


path_quality = "results/stats/pkpd-quality/opti/quality_vs_covariables.csv"
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
        
# Create exploratory visualizations
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS: MAE vs Covariates")
print("="*70 + "\n")

plot_mae_vs_quantitative_covariates(df, quantitative_covariates)
plot_mae_vs_binary_covariates(df, binary_covariates)

print("\n" + "="*70)
print("STATISTICAL MODELING")
print("="*70 + "\n")


# Define models to test
to_test_model_formulae = [
    "mae ~ 1",
    "mae ~ DFG",
    "mae ~ HTA",
    "mae ~ IECARA",
    "mae ~ age",
    "mae ~ sex",
    "mae ~ bmi",
    "mae ~ age + bmi + DFG + HTA + IECARA",
]

names = [1] * len(to_test_model_formulae)

link_function = sm.families.links.Log()  # None

output_dir = (
    "log-link"
    if type(link_function) is statsmodels.genmod.families.links.Log
    else "canonical-link"
)

# Run all models and collect results
models_results = []
alpha = 0.05  # Significance level (Type I error rate)

for formula, name in zip(to_test_model_formulae, names):
    print(f"\n{'#'*70}")
    print(f"# Processing Model: {formula}")
    print(f"{'#'*70}")

    result = complete_glm_model_analysis(
        dataframe=df,
        formula=formula,
        family=sm.families.Gamma(link=link_function),
        output=name,
        output_dir=output_dir,
        alpha=alpha
    )

    if result is not None:
        models_results.append(result)

# Save all successful models to JSON
if models_results:
    save_models_analysis(models_results, output_dir)
    print(f"\n✓ Pipeline complete: {len(models_results)}/{len(names)} models saved")
else:
    print("\n⚠ No models converged successfully. No results saved.")