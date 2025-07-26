import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_distribution_similarity(df, column, platform_col='platform_name'):
    """Calculate similarity between platform distributions using cosine distance."""
    
    # Filter out rows with missing values in the target column
    df_clean = df.dropna(subset=[column])
    
    if len(df_clean) == 0:
        raise ValueError(f"No valid data found for column '{column}'")
    
    # Create pivot table with proportions
    pivot_data = (df_clean.groupby([platform_col, column])
                  .size()
                  .unstack(fill_value=0)
                  .div(df_clean.groupby(platform_col).size(), axis=0))
    
    # Remove platforms with all zeros (no data)
    pivot_data = pivot_data.loc[pivot_data.sum(axis=1) > 0]
    
    if len(pivot_data) < 2:
        raise ValueError(f"Need at least 2 platforms with data for column '{column}', found {len(pivot_data)}")
    
    # Calculate pairwise cosine distances
    distances = pdist(pivot_data.values, metric='cosine')
    
    # Handle NaN values in distances (identical distributions)
    distances = np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=0.0)
    
    distance_matrix = squareform(distances)
    
    # Create similarity matrix (1 - distance)
    similarity_matrix = 1 - distance_matrix
    
    # Convert to DataFrame for easier handling
    similarity_df = pd.DataFrame(similarity_matrix, 
                                index=pivot_data.index, 
                                columns=pivot_data.index)
    
    return similarity_df, pivot_data

def get_similarity_order(similarity_df):
    """Get platform ordering based on hierarchical clustering of similarity."""
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_df.values
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Get the order from dendrogram
    dend = dendrogram(linkage_matrix, labels=similarity_df.index, no_plot=True)
    ordered_platforms = [similarity_df.index[i] for i in dend['leaves']]
    
    return ordered_platforms

def plot_facets_sorted(df, column, clean_prefix=None, platform_col='platform_name'):
    """Plot facets sorted by distribution similarity."""
    
    # Get similarity ordering
    similarity_df, pivot_data = calculate_distribution_similarity(df, column, platform_col)
    ordered_platforms = get_similarity_order(similarity_df)

    
    # Prepare data
    cat_data = (df.groupby([platform_col, column])
                .size()
                .reset_index(name='count'))
    
    # Clean column values if prefix specified
    if clean_prefix:
        cat_data[column] = cat_data[column].str.replace(clean_prefix, "")
    
    # Calculate proportions within each platform
    cat_data['total'] = cat_data.groupby(platform_col)['count'].transform('sum')
    cat_data['proportion'] = cat_data['count'] / cat_data['total']
    
    # Convert platform_name to categorical with our custom order
    cat_data[platform_col] = pd.Categorical(cat_data[platform_col], 
                                           categories=ordered_platforms, 
                                           ordered=True)
    
    # Create facet plot with one plot per row and shared x-axis
    g = sns.FacetGrid(cat_data, row=platform_col, height=2, aspect=4, sharex=True)
    g.map(plt.bar, column, 'proportion')
    g.set_titles('{row_name}')
    
    # Set x-axis labels on all subplots
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel(column.replace('_', ' ').title())
    
    plt.suptitle(f'Platform {column.replace("_", " ").title()} Distributions (Sorted by Similarity)', y=1.02)
    plt.show()
    
    return similarity_df

def plot_heatmap_sorted(df, column, clean_prefix=None, platform_col='platform_name', ordered_platforms=None):
    """Plot heatmap sorted by platform similarity."""
    
    # Get ordering if not provided
    if ordered_platforms is None:
        similarity_df, _ = calculate_distribution_similarity(df, column, platform_col)
        ordered_platforms = get_similarity_order(similarity_df)
    
    # Create pivot table
    pivot_data = (df.groupby([platform_col, column])
                  .size()
                  .unstack(fill_value=0)
                  .div(df.groupby(platform_col).size(), axis=0))
    
    # Clean column names if prefix specified
    if clean_prefix:
        pivot_data.columns = pivot_data.columns.str.replace(clean_prefix, "")
    
    # Reorder rows according to similarity
    pivot_data = pivot_data.reindex(ordered_platforms)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title(f"{column.replace('_', ' ').title()} Distribution by Platform (Sorted by Similarity)")
    plt.ylabel("Platform")
    plt.xlabel(column.replace('_', ' ').title())
    plt.tight_layout()
    plt.show()


def show_hist(df_relevant, col, clean_prefix=None, top_n=None):
    counts = df_relevant[col].value_counts(normalize=True).sort_values(ascending=True).reset_index().assign(col=lambda x: x[col].str.replace(clean_prefix, "", regex=False) if clean_prefix else x[col])
    if top_n is not None:
        counts = counts.head(top_n)
    # Count occurrences and calculate proportions
    counts["proportion"] = counts["proportion"] / counts["proportion"].sum()

    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(counts[col], counts["proportion"])
    plt.xlabel("Proportion")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()



# Convert string representations of lists to actual lists if needed
# (in case the lists are stored as strings)
def safe_eval_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            # If it's a plain string that's not a list representation, wrap it in a list
            return [x]
    if isinstance(x, list):
        return x
    # For any other type, convert to string and wrap in list
    return [str(x)]

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.stats.contingency_tables as sm_ct
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns

def analyze_categorical_association(df, platform_col='platform_name', 
                                  var1='category', var2='decision_visibility',
                                  alpha=0.05, min_expected_freq=5, verbose=False):
    """
    Comprehensive analysis of association between two categorical variables per platform.
    """
    
    results = []
    detailed_results = {}
    if verbose:
        print(f"Analyzing association between '{var1}' and '{var2}' by platform\n")
        print("=" * 80)
        
    # Get unique platforms
    platforms = df[platform_col].unique()
    
    for platform in platforms:

        if verbose:
            print(f"\nPlatform: {platform}")
            print("-" * 40)

        # Filter data for current platform
        platform_data = df[df[platform_col] == platform]
        
        # Create contingency table
        cont_table = pd.crosstab(platform_data[var1], platform_data[var2])
        if verbose:
            print(f"Contingency table shape: {cont_table.shape}")
            print(f"Total observations: {cont_table.sum().sum()}")

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(cont_table)
        
        # Check assumptions
        min_expected = expected.min()
        cells_below_5 = (expected < 5).sum()
        total_cells = expected.size
        pct_cells_below_5 = (cells_below_5 / total_cells) * 100

        if verbose:
            print(f"Minimum expected frequency: {min_expected:.2f}")
            print(f"Cells with expected < 5: {cells_below_5}/{total_cells} ({pct_cells_below_5:.1f}%)")

        # Decide on test based on assumptions
        test_used = "chi-square"
        if min_expected < min_expected_freq or pct_cells_below_5 > 20:
            print(f"Warning: Chi-square assumptions violated!")
            if cont_table.shape == (2, 2):
                # Use Fisher's exact test for 2x2 tables
                oddsratio, p_value = fisher_exact(cont_table)
                test_used = "fisher"
            
            else:
                # Use Monte Carlo simulation for larger tables
                chi2, p_value, dof, expected = chi2_contingency(cont_table, lambda_=None)
                test_used = "chi-square (Monte Carlo)"
            
        
        # Calculate effect size (Cramér's V)
        n = cont_table.sum().sum()
        min_dim = min(cont_table.shape[0] - 1, cont_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        # Calculate standardized residuals
        residuals = (cont_table - expected) / np.sqrt(expected)
        
        # Store results
        result = {
            'platform': platform,
            'n_observations': n,
            'n_categories_var1': cont_table.shape[0],
            'n_categories_var2': cont_table.shape[1],
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'test_used': test_used,
            'min_expected_freq': min_expected,
            'pct_cells_below_5': pct_cells_below_5,
            'significant': p_value < alpha
        }
        results.append(result)
        
        # Store detailed results
        detailed_results[platform] = {
            'contingency_table': cont_table,
            'expected': expected,
            'residuals': residuals,
            'significant_cells': np.abs(residuals) > 2
        }
        if verbose:
            print(f"\nResults for platform '{platform}':")
            print(f"Chi-square statistic: {chi2:.3f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Cramér's V: {cramers_v:.3f}")
            print(f"Test used: {test_used}")
            print(f"Significant at α={alpha}: {'Yes' if p_value < alpha else 'No'}")

    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    if verbose:
        print("\n" + "=" * 80)
        print("Multiple Testing Correction (Bonferroni)")
        print("=" * 80)


    p_values = results_df['p_value'].values
    corrected = multipletests(p_values, method='bonferroni', alpha=alpha)
    results_df['p_value_corrected'] = corrected[1]
    results_df['significant_corrected'] = corrected[0]
    
    print(f"\nSignificant associations after correction:")
    sig_platforms = results_df[results_df['significant_corrected']]['platform'].tolist()
    if sig_platforms:
        for platform in sig_platforms:
            row = results_df[results_df['platform'] == platform].iloc[0]
            print(f"- {platform}: p={row['p_value']:.4f}, "
                  f"p_corrected={row['p_value_corrected']:.4f}, "
                  f"Cramér's V={row['cramers_v']:.3f}")
    else:
        print("No significant associations found after correction.")
    
    return results_df, detailed_results


def print_summary_statistics(results_df):
    """
    Print summary statistics of the analysis.
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal platforms analyzed: {len(results_df)}")
    print(f"Significant associations (uncorrected): {results_df['significant'].sum()}")
    print(f"Significant associations (corrected): {results_df['significant_corrected'].sum()}")
    
    print("\nEffect sizes (Cramér's V) interpretation:")
    print("- Small: 0.1 - 0.3")
    print("- Medium: 0.3 - 0.5")
    print("- Large: > 0.5")
    
    print("\nEffect size distribution:")
    results_df['effect_size_category'] = pd.cut(results_df['cramers_v'], 
                                                bins=[0, 0.1, 0.3, 0.5, 1.0],
                                                labels=['Negligible', 'Small', 'Medium', 'Large'])
    print(results_df['effect_size_category'].value_counts().sort_index())


def visualize_associations(df, results_df, detailed_results, 
                          var1='category', var2='decision_visibility',
                          platform_col='platform_name', show_all=False):

    
    # Determine which platforms to visualize
    if show_all:
        platforms_to_plot = results_df['platform'].tolist()
    else:
        platforms_to_plot = results_df[results_df['significant_corrected']]['platform'].tolist()
    
    if not platforms_to_plot:
        print("\nNo significant associations to visualize.")
        return
    

    # 2. Create individual grouped bar charts for each platform
    print("\nGenerating distribution bar charts...")
    for platform in platforms_to_plot:
        platform_data = df[df[platform_col] == platform]
        
        # Create percentage crosstab - switched var1 and var2
        cont_table = pd.crosstab(platform_data[var2], platform_data[var1], normalize='index') * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Choose color palette based on number of categories
        n_categories = cont_table.shape[1]
        
        # Good categorical color palettes
        if n_categories <= 8:
            # Use ColorBrewer Set2 for up to 8 categories - great for categorical data
            colors = plt.cm.Set2(np.linspace(0, 1, n_categories))
        elif n_categories <= 12:
            # Use Paired colormap for up to 12 categories
            colors = plt.cm.Paired(np.linspace(0, 1, n_categories))
        else:
            # Use tab20 for up to 20 categories
            colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
        
        # Alternative: Use seaborn color palettes
        # colors = sns.color_palette("husl", n_categories)  # Good for many categories
        # colors = sns.color_palette("Set2", n_categories)  # Muted, good for 8 or fewer
        # colors = sns.color_palette("tab10", n_categories)  # Tableau colors
        
        cont_table.plot(kind='bar', ax=ax, width=0.8, color=colors)
        
        row = results_df[results_df['platform'] == platform].iloc[0]
        
        ax.set_title(f'Distribution of {var1} by {var2}\nPlatform: {platform}\n'
                    f'(n_observations = {row["n_observations"]:,d})',
                    fontsize=14, pad=20)
        ax.set_xlabel(var2, fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.legend(title=var1, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()    
    
def plot_grouped_bars_by_platform(df, x_col, color_col, platform_col='platform_name', clean_prefix=None, top_n_color=20):

    
    # Get unique platforms
    platforms = sorted(df[platform_col].unique())
    
    print(f"Creating plots for {len(platforms)} platforms...")
    print(f"X-axis: {x_col} | Color: {color_col} (top {top_n_color} per platform)")
    print("=" * 60)
    if clean_prefix:
        df[x_col] = df[x_col].str.replace(clean_prefix, "")

    # Plot for each platform separately
    for platform in platforms:
        try:
            # Filter data for this platform
            platform_data = df[df[platform_col] == platform].copy()
            print(f"Processing platform: {platform} with {platform_data.shape=}")
            if len(platform_data) == 0:
                print(f"⚠️  {platform}: No data available")
                continue
            
            # Get top N categories for color column FOR THIS PLATFORM
            top_categories = platform_data[color_col].value_counts().head(top_n_color).index.tolist()

            # Replace non-top categories with 'Other' FOR THIS PLATFORM
            platform_data[f'{color_col}_grouped'] = platform_data[color_col].apply(
                lambda x: x if x in top_categories else 'Other'
            )
            
            # Create pivot table for grouped bars
            pivot_data = platform_data.groupby([x_col, f'{color_col}_grouped']).size().unstack(fill_value=0)
            
            # Create individual plot with grouped bars
            plt.figure(figsize=(14, 6))
            ax = pivot_data.plot(kind='bar', ax=plt.gca(), colormap='tab20')
            
            # Customize plot
            plt.title(f"{platform}\n{x_col.replace('_', ' ').title()} by {color_col.replace('_', ' ').title()}", 
                    fontsize=14, fontweight='bold')
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # Add grid and styling
            plt.grid(axis='y', alpha=0.3)
            plt.legend(title=color_col.replace('_', ' ').title(), 
                    bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add data summary
            plt.text(0.02, 0.98, f'n = {len(platform_data)}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')
            
            plt.tight_layout()
            plt.show()
            
            print(f"✅ {platform}: {len(platform_data)} records, {len(top_categories)} top categories")
        except Exception as e:
            print(f"❌ Error processing platform {platform}: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"- Total platforms plotted: {len(platforms)}")
    print(f"- X-axis categories: {df[x_col].nunique()}")
    print(f"- Top {top_n_color} categories calculated per platform individually")
    print(f"- Total records: {len(df)}")
