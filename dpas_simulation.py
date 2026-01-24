import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION
# =====================================================
CONFIG = {
    'chunk_size': 10000,           # Process data in chunks
    'C': 100.0,                    # Total bandwidth (Mbps)
    'alpha': 0.4,                  # DPAS learning rate
    'min_bandwidth': 5,            # Minimum slice bandwidth
    'max_bandwidth': 80,           # Maximum slice bandwidth
    'sla_thresholds': [100, 10, 500],  # ms for eMBB, URLLC, mMTC
    'required_cols': ['datetime', 'smsin', 'smsout', 'callin', 'callout', 'internet']
}

# =====================================================
# OPTIMIZED DATA LOADING
# =====================================================
def load_excel_optimized(filepath, chunk_size=None):
    """Load Excel with memory optimization"""
    try:
        # Read with specific dtypes to reduce memory
        dtype_dict = {
            'smsin': 'float32',
            'smsout': 'float32',
            'callin': 'float32',
            'callout': 'float32',
            'internet': 'float32'
        }

        if chunk_size:
            # For very large files, use chunked reading
            chunks = []
            for chunk in pd.read_excel(filepath, chunksize=chunk_size, dtype=dtype_dict):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            df = pd.read_excel(filepath, dtype=dtype_dict)

        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# =====================================================
# DATA CLEANING PIPELINE
# =====================================================
def clean_dataframe(df):
    """Clean and prepare dataframe efficiently"""
    # Remove empty columns
    df = df.dropna(axis=1, how='all')

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Check required columns
    missing = [c for c in CONFIG['required_cols'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only needed columns
    df = df[CONFIG['required_cols']].copy()

    # Convert to numeric efficiently
    for col in CONFIG['required_cols'][1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN with 0
    df.fillna(0, inplace=True)

    # Convert to float32 to save memory
    for col in df.columns[1:]:
        df[col] = df[col].astype('float32')

    return df

# =====================================================
# FEATURE ENGINEERING (VECTORIZED)
# =====================================================
def engineer_features(df):
    """Vectorized feature creation"""
    df['rt'] = df['callin'] + df['callout']
    df['sig'] = df['smsin'] + df['smsout']
    df['data'] = df['internet']
    df['total'] = df['rt'] + df['sig'] + df['data']

    # Remove zero traffic rows
    df = df[df['total'] > 0].copy()

    # Calculate ratios (vectorized)
    df['rt_ratio'] = df['rt'] / df['total']
    df['sig_ratio'] = df['sig'] / df['total']
    df['data_ratio'] = df['data'] / df['total']

    return df

# =====================================================
# TRAFFIC CLASSIFICATION (VECTORIZED)
# =====================================================
def classify_traffic_vectorized(df):
    """Vectorized traffic classification"""
    conditions = [
        df['data_ratio'] >= 0.6,
        df['rt_ratio'] >= 0.3,
        df['sig_ratio'] >= 0.2
    ]
    choices = ['eMBB', 'URLLC', 'mMTC']
    df['Traffic_Class'] = np.select(conditions, choices, default='Mixed')
    return df

# =====================================================
# DPAS ALGORITHM (OPTIMIZED)
# =====================================================
def dpas_slicing(demands, C=100.0, alpha=0.4, min_bw=5, max_bw=80):
    """
    Optimized DPAS with NumPy vectorization

    Args:
        demands: (N, 3) array of [data, rt, sig] demands
        C: Total bandwidth
        alpha: Learning rate
        min_bw, max_bw: Bandwidth constraints

    Returns:
        B_hist: (N, 3) array of bandwidth allocations
    """
    N = len(demands)
    B_hist = np.zeros((N, 3), dtype=np.float32)
    B = np.array([C/3, C/3, C/3], dtype=np.float32)

    for i in range(N):
        D = demands[i]
        pressure = D / np.maximum(B, 1e-6)
        delta = pressure - pressure.mean()
        B = B - alpha * delta * B
        B = np.clip(B, min_bw, max_bw)
        B = B * (C / B.sum())
        B_hist[i] = B

    return B_hist

# =====================================================
# BATCH PROCESSING FOR MULTIPLE FILES
# =====================================================
def process_multiple_files(file_paths, output_dir='results'):
    """Process multiple Excel files in batch"""
    Path(output_dir).mkdir(exist_ok=True)
    results = []

    print(f"Processing {len(file_paths)} files...")

    for filepath in tqdm(file_paths):
        try:
            # Load and clean
            df = load_excel_optimized(filepath)
            if df is None:
                continue

            df = clean_dataframe(df)
            df = engineer_features(df)
            df = classify_traffic_vectorized(df)

            # DPAS slicing
            demands = df[['data', 'rt', 'sig']].values.astype(np.float32)
            B_hist = dpas_slicing(
                demands,
                CONFIG['C'],
                CONFIG['alpha'],
                CONFIG['min_bandwidth'],
                CONFIG['max_bandwidth']
            )

            # Calculate SLA
            delay = demands / np.maximum(B_hist, 1e-6)
            sla_thresholds = np.array(CONFIG['sla_thresholds'])
            violations = delay > sla_thresholds
            sla_rate = 100 * (1 - violations.mean())

            # Store results
            result = {
                'file': Path(filepath).name,
                'n_records': len(df),
                'sla_rate': sla_rate,
                'avg_embb_bw': B_hist[:, 0].mean(),
                'avg_urllc_bw': B_hist[:, 1].mean(),
                'avg_mmtc_bw': B_hist[:, 2].mean(),
                'traffic_dist': df['Traffic_Class'].value_counts().to_dict()
            }
            results.append(result)

            # Save processed data
            output_file = Path(output_dir) / f"{Path(filepath).stem}_processed.parquet"
            df.to_parquet(output_file, index=False)

            # Save bandwidth history
            bw_df = pd.DataFrame(B_hist, columns=['eMBB', 'URLLC', 'mMTC'])
            bw_file = Path(output_dir) / f"{Path(filepath).stem}_bandwidth.parquet"
            bw_df.to_parquet(bw_file, index=False)

            # Clean up memory
            del df, demands, B_hist, delay
            gc.collect()

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    return pd.DataFrame(results)

# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================
def plot_traffic_composition(df, max_points=5000):
    """Plot traffic composition with downsampling for large datasets"""
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step]
    else:
        df_plot = df

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        range(len(df_plot)),
        df_plot['data'],
        df_plot['rt'],
        df_plot['sig'],
        labels=['eMBB (Internet)', 'URLLC (Calls)', 'mMTC (SMS)'],
        alpha=0.8
    )
    ax.legend(loc='upper right')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Traffic Volume')
    ax.set_title('Traffic Composition Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_bandwidth_allocation(B_hist, max_points=5000):
    """Plot bandwidth allocation with downsampling"""
    if len(B_hist) > max_points:
        step = len(B_hist) // max_points
        B_plot = B_hist[::step]
    else:
        B_plot = B_hist

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(B_plot[:, 0], label='eMBB', linewidth=1.5)
    ax.plot(B_plot[:, 1], label='URLLC', linewidth=1.5)
    ax.plot(B_plot[:, 2], label='mMTC', linewidth=1.5)
    ax.set_ylabel('Bandwidth (Mbps)')
    ax.set_xlabel('Time Index')
    ax.set_title('DPAS Bandwidth Allocation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_summary_stats(results_df):
    """Plot summary statistics for multiple files"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SLA rates
    axes[0, 0].bar(range(len(results_df)), results_df['sla_rate'])
    axes[0, 0].set_ylabel('SLA Rate (%)')
    axes[0, 0].set_title('SLA Satisfaction by File')
    axes[0, 0].grid(True, alpha=0.3)

    # Average bandwidths
    bw_data = results_df[['avg_embb_bw', 'avg_urllc_bw', 'avg_mmtc_bw']]
    bw_data.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_ylabel('Bandwidth (Mbps)')
    axes[0, 1].set_title('Average Bandwidth Allocation')
    axes[0, 1].legend(['eMBB', 'URLLC', 'mMTC'])
    axes[0, 1].grid(True, alpha=0.3)

    # Record counts
    axes[1, 0].bar(range(len(results_df)), results_df['n_records'])
    axes[1, 0].set_ylabel('Number of Records')
    axes[1, 0].set_title('Dataset Sizes')
    axes[1, 0].grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[1, 1])

    plt.tight_layout()
    return fig

# =====================================================
# MAIN EXECUTION FOR COLAB
# =====================================================
def main():
    """Main execution function for Colab"""
    print("=" * 60)
    print("OPTIMIZED NETWORK SLICING FOR GOOGLE COLAB")
    print("=" * 60)

    # Upload files in Colab
    from google.colab import files
    print("\n📤 Upload your Excel files...")
    uploaded = files.upload()

    if not uploaded:
        print("❌ No files uploaded!")
        return

    file_paths = list(uploaded.keys())
    print(f"\n✅ Uploaded {len(file_paths)} file(s)")

    # Process files
    results_df = process_multiple_files(file_paths)

    # Display results
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save summary
    results_df.to_csv('processing_summary.csv', index=False)
    print("\n💾 Summary saved to: processing_summary.csv")

    # Generate summary plots
    if len(results_df) > 0:
        fig = plot_summary_stats(results_df)
        plt.savefig('summary_stats.png', dpi=150, bbox_inches='tight')
        print("📊 Summary plot saved to: summary_stats.png")
        plt.show()

    # Process first file for detailed visualization
    if len(file_paths) > 0:
        print(f"\n📈 Generating detailed plots for: {file_paths[0]}")
        df = load_excel_optimized(file_paths[0])
        df = clean_dataframe(df)
        df = engineer_features(df)
        df = classify_traffic_vectorized(df)

        # Traffic composition
        fig1 = plot_traffic_composition(df)
        plt.savefig('traffic_composition.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Bandwidth allocation
        demands = df[['data', 'rt', 'sig']].values.astype(np.float32)
        B_hist = dpas_slicing(demands, CONFIG['C'], CONFIG['alpha'])
        fig2 = plot_bandwidth_allocation(B_hist)
        plt.savefig('bandwidth_allocation.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\n✅ Processing complete!")
        print(f"📁 Results saved in 'results/' directory")

        # Download results
        print("\n💾 Downloading results...")
        files.download('processing_summary.csv')

# =====================================================
# RUN IN COLAB
# =====================================================
if __name__ == "__main__":
    main()