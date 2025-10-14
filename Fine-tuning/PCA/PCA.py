import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from typing import Tuple

# ==========================================================================================
# --- 1. Global Style Settings (for Publication) ---
# ==========================================================================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.unicode_minus'] = False

# ==========================================================================================
# --- 2. PLOTTING FUNCTION ---
# ==========================================================================================

def generate_single_pca_plot(df: pd.DataFrame, state: str, filename: str, xlims: Tuple[float, float], ylims: Tuple[float, float]):
    """
    Generates an independent, publication-quality PCA plot for a given state ('before' or 'after').
    Uses xlims and ylims parameters to unify axes across plots for better comparability.
    """
    print(f"\nGenerating publication-quality plot for '{state.capitalize()} Fine-tuning' state...")

    # Define color palettes and markers for different data sources and types
    base_colors = {
        'CSN Train': ('#1f77b4', '#aec7e8'),
        'CSN Test': ('#ff7f0e', '#ffbb78'),
        'CoSQA': ('#2ca02c', '#98df8a')
    }
    plot_palette = {f'{source} - {type}': color for source, (dark, light) in base_colors.items() for type, color in [('Code', dark), ('Query', light)]}
    markers = {'Query': 'o', 'Code': 'X'}

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        x=f'pca1_{state}', y=f'pca2_{state}',
        hue='label', style='type', markers=markers,
        palette=plot_palette, data=df, s=60,
        alpha=0.85, edgecolor='w', linewidth=0.3, ax=ax,
        legend=True
    )

    # Draw connecting lines between query-code pairs
    for pair_id in df['pair_id'].unique():
        pair_df = df[df['pair_id'] == pair_id]
        query_point = pair_df[pair_df['type'] == 'Query']
        code_point = pair_df[pair_df['type'] == 'Code']
        if not query_point.empty and not code_point.empty:
            point_label = query_point['label'].values[0]
            line_color = plot_palette.get(point_label, 'grey')

            ax.plot(
                [query_point[f'pca1_{state}'].values[0], code_point[f'pca1_{state}'].values[0]],
                [query_point[f'pca2_{state}'].values[0], code_point[f'pca2_{state}'].values[0]],
                color=line_color,
                linestyle='--', linewidth=0.8, alpha=0.6
            )

    # Apply the global axis limits for consistency
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)

    # Adjust legend position, font size, and transparency
    ax.legend(title='Data & Type', loc='upper left', fontsize=7, framealpha=0.7)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[Figure] Plot saved to {filename}")
    plt.close(fig)

# ==========================================================================================
# --- 3. MAIN VISUALIZATION WORKFLOW ---
# ==========================================================================================
def main(args):
    """
    Main function to run the PCA visualization workflow.
    """
    # Define cache directory based on output directory
    CACHE_DIR = os.path.join(args.output_dir, "cache")
    
    print(f"--- Loading selection data from: {args.samples_file} ---")

    try:
        with open(args.samples_file, 'r', encoding='utf-8') as f:
            selected_samples = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Selection file not found at '{args.samples_file}'.")
        return

    print(f"Found {len(selected_samples)} samples to visualize.")

    print("\n--- Loading all cached embeddings ---")
    try:
        q_embeds_orig = np.load(os.path.join(CACHE_DIR, 'embeds_original_queries.npy'))
        c_embeds_orig = np.load(os.path.join(CACHE_DIR, 'embeds_original_codes.npy'))
        q_embeds_ft = np.load(os.path.join(CACHE_DIR, 'embeds_finetuned_queries.npy'))
        c_embeds_ft = np.load(os.path.join(CACHE_DIR, 'embeds_finetuned_codes.npy'))
    except FileNotFoundError as e:
        print(f"Error: Cached embedding file not found in '{CACHE_DIR}'.")
        print(f"Details: {e}")
        print("Please ensure the embedding generation script has been run and data is in the correct location.")
        return

    # Validate that the number of samples matches the number of embeddings
    if not (len(selected_samples) == len(q_embeds_orig) == len(c_embeds_orig) == len(q_embeds_ft) == len(c_embeds_ft)):
        print("Error: The number of samples in the JSONL file does not match the number of embeddings in .npy files.")
        print(f"JSONL samples: {len(selected_samples)}, Query Embeddings: {len(q_embeds_orig)}")
        return

    print("\n--- Fitting PCA model on original data ---")
    pca = PCA(n_components=2)
    all_embeds_for_pca_fit = np.vstack((q_embeds_orig, c_embeds_orig))
    pca.fit(all_embeds_for_pca_fit)

    # Transform embeddings using the fitted PCA model
    q_pca_orig, c_pca_orig = pca.transform(q_embeds_orig), pca.transform(c_embeds_orig)
    q_pca_ft, c_pca_ft = pca.transform(q_embeds_ft), pca.transform(c_embeds_ft)

    print("\n--- Preparing data for the plot ---")
    plot_data = []
    for i, sample in enumerate(selected_samples):
        source = sample['source']
        pair_id = sample['pair_id']

        plot_data.append({
            'pair_id': pair_id, 'source': source, 'type': 'Query', 'label': f'{source} - Query',
            'pca1_before': q_pca_orig[i, 0], 'pca2_before': q_pca_orig[i, 1],
            'pca1_after': q_pca_ft[i, 0], 'pca2_after': q_pca_ft[i, 1]
        })
        plot_data.append({
            'pair_id': pair_id, 'source': source, 'type': 'Code', 'label': f'{source} - Code',
            'pca1_before': c_pca_orig[i, 0], 'pca2_before': c_pca_orig[i, 1],
            'pca1_after': c_pca_ft[i, 0], 'pca2_after': c_pca_ft[i, 1]
        })

    df_plot = pd.DataFrame(plot_data)

    # Calculate global axis limits to ensure both plots are comparable
    print("\n--- Calculating global axis limits for consistent plotting ---")
    pca1_min = min(df_plot['pca1_before'].min(), df_plot['pca1_after'].min())
    pca1_max = max(df_plot['pca1_before'].max(), df_plot['pca1_after'].max())
    pca2_min = min(df_plot['pca2_before'].min(), df_plot['pca2_after'].min())
    pca2_max = max(df_plot['pca2_before'].max(), df_plot['pca2_after'].max())

    x_range = pca1_max - pca1_min
    y_range = pca2_max - pca2_min
    xlims = (pca1_min - 0.05 * x_range, pca1_max + 0.05 * x_range)
    ylims = (pca2_min - 0.05 * y_range, pca2_max + 0.05 * y_range)

    print(f"Global X-axis limits (PC1): {xlims}")
    print(f"Global Y-axis limits (PC2): {ylims}")

    # Generate and save the plots
    strategy_name = os.path.basename(args.samples_file).split('_')[-1].replace('.jsonl', '')

    filename_before = os.path.join(args.output_dir, f"figure_pca_before_finetuning_{strategy_name}.pdf")
    generate_single_pca_plot(df_plot, 'before', filename_before, xlims, ylims)

    filename_after = os.path.join(args.output_dir, f"figure_pca_after_finetuning_{strategy_name}.pdf")
    generate_single_pca_plot(df_plot, 'after', filename_after, xlims, ylims)

    print(f"\n✅ Visualization complete. Two separate plots with consistent axes have been generated in '{args.output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PCA visualizations for code/query embeddings before and after fine-tuning.")
    parser.add_argument("--samples_file", type=str, required=True, help="Path to the JSONL file containing selected samples for visualization.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output plots. A 'cache' subdirectory with embeddings is expected inside.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)
