import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import re

def format_sparsity_coeff(value: Any) -> str:
    """Format sparsity coefficient with appropriate precision."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    elif value >= 1.0:
        return f'{value:.1f}'  # e.g., "50.0", "3.0"
    elif value >= 0.01:
        return f'{value:.2f}'  # e.g., "0.05", "0.10"
    elif value >= 0.001:
        return f'{value:.3f}'  # e.g., "0.005", "0.001"
    else:
        return f'{value:.0e}'  # e.g., "5e-05", "1e-06"

def extract_multi_layer_metrics(run: Any) -> Dict[str, Any]:
    """Extract metrics for runs with SAEs at multiple layers."""
    run_data = {
        'name': run.name,
        'id': run.id,
        'state': run.state
    }
    
    if run.state != "finished":
        return run_data
    
    # Determine run type based on name
    if run.name.startswith("ReLU"):
        run_data['method'] = "ReLU"
    elif run.name.startswith("Variational"):
        run_data['method'] = "Variational"
    else:
        run_data['method'] = "Unknown"
    
    # Extract sparsity coefficient from config
    try:
        sparsity_coeff = run.config['loss']['sparsity']['coeff']
        run_data['sparsity_coeff'] = sparsity_coeff
    except (KeyError, TypeError):
        # Try alternative path for backwards compatibility
        try:
            sparsity_coeff = run.config['loss']['sparsity']['final_coeff']
            run_data['sparsity_coeff'] = sparsity_coeff
        except (KeyError, TypeError):
            run_data['sparsity_coeff'] = None
    
    # Extract layers from metrics
    layers = set()
    sparsity_metrics = {}
    recon_loss_metrics = {}
    
    for key, value in run.summary_metrics.items():
        # Look for sparsity metrics: "sparsity/eval/L_0/blocks.{layer}.hook_resid_pre"
        sparsity_match = re.search(r'sparsity/eval/L_0/blocks\.(\d+)\.hook_resid_pre', key)
        if sparsity_match:
            layer = int(sparsity_match.group(1))
            layers.add(layer)
            sparsity_metrics[layer] = value
        
        # Look for reconstruction loss: "loss/eval/out_to_in/blocks.{layer}.hook_resid_pre"
        # Note: we also check in_to_orig as the pattern might be different
        recon_match = re.search(r'loss/eval/(?:out_to_in|in_to_orig)/blocks\.(\d+)\.hook_resid_(?:pre|post)', key)
        if recon_match:
            layer = int(recon_match.group(1))
            if 'out_to_in' in key or layer not in recon_loss_metrics:  # Prefer out_to_in over in_to_orig
                layers.add(layer)
                recon_loss_metrics[layer] = value
    
    # Store per-layer metrics
    for layer in sorted(layers):
        run_data[f'sparsity_layer_{layer}'] = sparsity_metrics.get(layer, np.nan)
        run_data[f'recon_loss_layer_{layer}'] = recon_loss_metrics.get(layer, np.nan)
    
    run_data['layers'] = sorted(layers)
    
    return run_data

def create_pareto_plot(df: pd.DataFrame, layers: List[int], save_path: str = None):
    """Create a pareto frontier plot for sparsity vs reconstruction loss."""
    
    # Set up the plot
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    colors = {'ReLU': '#d62728', 'Variational': '#2ca02c'}  # Red for ReLU, Green for Variational
    markers = {'ReLU': 'o', 'Variational': 's'}  # Circle for ReLU, Square for Variational
    
    for i, layer in enumerate(layers):
        ax = axes[i]
        
        # Get data for this layer
        sparsity_col = f'sparsity_layer_{layer}'
        recon_col = f'recon_loss_layer_{layer}'
        
        # Filter out runs that don't have this layer
        layer_df = df.dropna(subset=[sparsity_col, recon_col])
        
        if layer_df.empty:
            ax.set_title(f'Layer {layer} (No data)')
            continue
        
        # Plot each method
        for method in ['ReLU', 'Variational']:
            method_df = layer_df[layer_df['method'] == method]
            if not method_df.empty:
                ax.scatter(
                    method_df[sparsity_col], 
                    method_df[recon_col],
                    c=colors[method], 
                    marker=markers[method],
                    s=60,
                    alpha=0.7,
                    label=method,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                # Add sparsity coefficient annotations
                for idx, row in method_df.iterrows():
                    if pd.notna(row['sparsity_coeff']):
                        ax.annotate(
                            f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                            (row[sparsity_col], row[recon_col]),
                            xytext=(5, 5),  # offset from point
                            textcoords='offset points',
                            fontsize=7,
                            color=colors[method],
                            alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                        )
        
        ax.set_xlabel('Sparsity (L0)')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title(f'Layer {layer}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set log scale if needed for better visualization
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig

# Setup wandb API
api = wandb.Api(api_key='b8fa6d3104a0f99ee8a99f7c7659b893559f1097')
sweep_project = "raymondl/gpt2-dict-size-10"
runs = api.runs(sweep_project)

print(f"Total runs: {len(runs)}")

# Extract data from all runs
print("Extracting metrics from runs...")
run_data_list = []
for run in runs:
    run_data = extract_multi_layer_metrics(run)
    run_data_list.append(run_data)

# Create DataFrame
df = pd.DataFrame(run_data_list)

# Filter to only finished runs
df = df[df['state'] == 'finished']
print(f"Finished runs: {len(df)}")

# Print summary
print("\n=== METHOD SUMMARY ===")
print(df['method'].value_counts())

# Find available layers
all_layers = set()
for _, row in df.iterrows():
    if 'layers' in row and isinstance(row['layers'], list):
        all_layers.update(row['layers'])

available_layers = sorted(all_layers)
print(f"\n=== AVAILABLE LAYERS ===")
print(f"Layers: {available_layers}")

# Print sample data for inspection
print("\n=== SAMPLE DATA ===")
if not df.empty:
    sample_run = df.iloc[0]
    print(f"Sample run: {sample_run['name']}")
    print(f"Method: {sample_run['method']}")
    print(f"Sparsity coeff: {sample_run.get('sparsity_coeff', 'N/A')}")
    for layer in available_layers:
        sparsity = sample_run.get(f'sparsity_layer_{layer}', 'N/A')
        recon = sample_run.get(f'recon_loss_layer_{layer}', 'N/A')
        print(f"  Layer {layer}: Sparsity={sparsity}, Recon Loss={recon}")
    
    print(f"\nSuccessfully extracted sparsity coefficients for {len(df)} runs")
    print(f"Sparsity coefficient range: {df['sparsity_coeff'].min():.0e} to {df['sparsity_coeff'].max():.1f}")
    
    # Print all runs with their IDs and sparsity coefficients
    print("\n=== ALL RUNS WITH IDs AND SPARSITY COEFFICIENTS ===")
    for method in ['ReLU', 'Variational']:
        method_df = df[df['method'] == method].sort_values('sparsity_coeff')
        print(f"\n{method} runs ({len(method_df)} total):")
        for idx, row in method_df.iterrows():
            sparsity_coeff_str = format_sparsity_coeff(row['sparsity_coeff'])
            print(f"  {row['id']} - λ={sparsity_coeff_str}")

# Create the pareto plot
if available_layers:
    # Focus on layers where SAEs are actually placed (have sparsity data)
    sae_layers = []
    for layer in available_layers:
        sparsity_col = f'sparsity_layer_{layer}'
        if sparsity_col in df.columns and not df[sparsity_col].isna().all():
            sae_layers.append(layer)
    
    print(f"\n=== SAE LAYERS WITH DATA ===")
    print(f"SAE layers: {sae_layers}")
    
    if sae_layers:
        print(f"\n=== CREATING PARETO PLOT ===")
        fig = create_pareto_plot(df, sae_layers, 'sparsity_vs_reconstruction_loss_pareto.png')
        
        # Also create a combined plot for SAE layers only
        print("\n=== CREATING COMBINED PLOT ===")
        
        plt.figure(figsize=(12, 8))
        
        colors = {'ReLU': '#d62728', 'Variational': '#2ca02c'}
        markers = {'ReLU': 'o', 'Variational': 's'}
        sizes = {2: 40, 4: 50, 6: 60, 8: 70, 10: 80}  # Different sizes for different layers
        
        legend_added = {'ReLU': False, 'Variational': False}
        
        for layer in sae_layers:
            sparsity_col = f'sparsity_layer_{layer}'
            recon_col = f'recon_loss_layer_{layer}'
            
            layer_df = df.dropna(subset=[sparsity_col, recon_col])
            
            for method in ['ReLU', 'Variational']:
                method_df = layer_df[layer_df['method'] == method]
                if not method_df.empty:
                    # Add label only once per method for legend
                    label = method if not legend_added[method] else ""
                    if label:
                        legend_added[method] = True
                    
                    plt.scatter(
                        method_df[sparsity_col], 
                        method_df[recon_col],
                        c=colors[method], 
                        marker=markers[method],
                        s=sizes.get(layer, 60),
                        alpha=0.7,
                        label=label,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    
                    # Add sparsity coefficient annotations
                    for idx, row in method_df.iterrows():
                        if pd.notna(row['sparsity_coeff']):
                            plt.annotate(
                                f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                                (row[sparsity_col], row[recon_col]),
                                xytext=(5, 5),  # offset from point
                                textcoords='offset points',
                                fontsize=8,
                                color=colors[method],
                                alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                            )
        
        plt.xlabel('Sparsity (L0)', fontsize=12)
        plt.ylabel('Reconstruction Loss', fontsize=12)
        plt.title('Sparsity vs Reconstruction Loss Trade-off\n(ReLU Baselines vs Variational SAEs)', fontsize=14)
        plt.yscale('log')
        
        # Only add legend if we have labeled artists
        if any(legend_added.values()):
            plt.legend(fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('combined_sparsity_vs_reconstruction_loss.png', dpi=300, bbox_inches='tight')
        plt.savefig('combined_sparsity_vs_reconstruction_loss.svg', bbox_inches='tight')
        plt.show()
        
        print("Combined plot saved to combined_sparsity_vs_reconstruction_loss.png")
        
        # Create a detailed per-layer comparison
        print("\n=== CREATING PER-LAYER COMPARISON ===")
        
        n_sae_layers = len(sae_layers)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, layer in enumerate(sae_layers):
            if i >= len(axes):
                break
                
            ax = axes[i]
            sparsity_col = f'sparsity_layer_{layer}'
            recon_col = f'recon_loss_layer_{layer}'
            
            layer_df = df.dropna(subset=[sparsity_col, recon_col])
            
            for method in ['ReLU', 'Variational']:
                method_df = layer_df[layer_df['method'] == method]
                if not method_df.empty:
                    ax.scatter(
                        method_df[sparsity_col], 
                        method_df[recon_col],
                        c=colors[method], 
                        marker=markers[method],
                        s=60,
                        alpha=0.7,
                        label=method,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    
                    # Add sparsity coefficient annotations
                    for idx, row in method_df.iterrows():
                        if pd.notna(row['sparsity_coeff']):
                            ax.annotate(
                                f'{format_sparsity_coeff(row["sparsity_coeff"])}',
                                (row[sparsity_col], row[recon_col]),
                                xytext=(5, 5),  # offset from point
                                textcoords='offset points',
                                fontsize=7,
                                color=colors[method],
                                alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                            )
            
            ax.set_xlabel('Sparsity (L0)')
            ax.set_ylabel('Reconstruction Loss')
            ax.set_title(f'Layer {layer}')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(sae_layers), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('per_layer_sparsity_vs_reconstruction_loss.png', dpi=300, bbox_inches='tight')
        plt.savefig('per_layer_sparsity_vs_reconstruction_loss.svg', bbox_inches='tight')
        plt.show()
        
        print("Per-layer plot saved to per_layer_sparsity_vs_reconstruction_loss.png")

else:
    print("No layers found with data!")

# Print detailed statistics for SAE layers only
print("\n=== DETAILED STATISTICS (SAE LAYERS ONLY) ===")
sae_layers = []
for layer in available_layers:
    sparsity_col = f'sparsity_layer_{layer}'
    if sparsity_col in df.columns and not df[sparsity_col].isna().all():
        sae_layers.append(layer)

for layer in sae_layers:
    sparsity_col = f'sparsity_layer_{layer}'
    recon_col = f'recon_loss_layer_{layer}'
    
    layer_df = df.dropna(subset=[sparsity_col, recon_col])
    
    print(f"\nLayer {layer}:")
    for method in ['ReLU', 'Variational']:
        method_df = layer_df[layer_df['method'] == method]
        if not method_df.empty:
            print(f"  {method}: {len(method_df)} runs")
            print(f"    Sparsity range: {method_df[sparsity_col].min():.2f} - {method_df[sparsity_col].max():.2f}")
            print(f"    Recon loss range: {method_df[recon_col].min():.2e} - {method_df[recon_col].max():.2e}")
            
            # Find best trade-offs (lowest sparsity for given reconstruction loss ranges)
            print(f"    Performance summary:")
            for recon_threshold in [0.001, 0.01, 0.1, 1.0]:
                viable_runs = method_df[method_df[recon_col] <= recon_threshold]
                if not viable_runs.empty:
                    min_sparsity = viable_runs[sparsity_col].min()
                    best_run = viable_runs[viable_runs[sparsity_col] == min_sparsity].iloc[0]
                    sparsity_coeff = format_sparsity_coeff(best_run['sparsity_coeff'])
                    print(f"      Best sparsity at recon ≤ {recon_threshold}: {min_sparsity:.2f} (run {best_run['id']}, λ={sparsity_coeff})")
