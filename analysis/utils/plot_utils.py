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


