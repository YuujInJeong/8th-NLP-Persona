import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

sns.set_theme(style="whitegrid")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--model_csv", type=str, required=True, help="Path to Model Result CSV")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name (e.g., 'Team1_Static')")
    parser.add_argument("--type", type=str, choices=['static', 'dynamic'], required=True, help="'static' or 'dynamic'")

    parser.add_argument("--steam_gt", type=str, default="datasets/ground_truth_steam.csv", help="Path to Steam GT")
    parser.add_argument("--stock_gt", type=str, default="datasets/ground_truth_stock.csv", help="Path to Stock GT")
    
    return parser.parse_args()

def load_ground_truth(path, value_col):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT File not found: {path}")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Date', value_col]].sort_values('Date')

def calculate_model_ratio(df_model, model_type):
    # YES/NO Parsing (Robust)
    df_model['Vote'] = df_model['Decision'].apply(lambda x: 1 if str(x).strip().upper().startswith('YES') else 0)
    
    if model_type == 'static':
        ratio = df_model['Vote'].mean()
        return ratio, None
    elif model_type == 'dynamic':
        if 'Simulation_Date' not in df_model.columns:
            raise ValueError("Dynamic model requires 'Simulation_Date' column.")
        df_model['Simulation_Date'] = pd.to_datetime(df_model['Simulation_Date'])
        daily_ratio = df_model.groupby('Simulation_Date')['Vote'].mean().reset_index()
        daily_ratio.columns = ['Date', 'Purchase_Ratio']
        return None, daily_ratio

def main():
    args = parse_args()
    print(f"--- Model Evaluation: {args.model_name} ({args.type}) ---")
    
    steam_gt = load_ground_truth(args.steam_gt, 'Positive_Ratio')
    stock_gt = load_ground_truth(args.stock_gt, 'Stock_Price')
    
    model_df = pd.read_csv(args.model_csv)
    static_ratio, dynamic_df = calculate_model_ratio(model_df, args.type)
    
    if args.type == 'static':
        merged_steam = steam_gt.copy()
        merged_steam['Model_Ratio'] = static_ratio
        merged_stock = stock_gt.copy()
        merged_stock['Model_Ratio'] = static_ratio
        print(f"   [Info] Static Ratio: {static_ratio:.4f}")
    else:
        merged_steam = pd.merge(steam_gt, dynamic_df, on='Date', how='inner')
        merged_steam.rename(columns={'Purchase_Ratio': 'Model_Ratio'}, inplace=True)
        merged_stock = pd.merge(stock_gt, dynamic_df, on='Date', how='inner')
        merged_stock.rename(columns={'Purchase_Ratio': 'Model_Ratio'}, inplace=True)
        print(f"   [Info] Matched Dates: {len(merged_steam)}")

    if len(merged_steam) < 2:
        print("Error: Not enough data points to calculate correlation.")
        return

    # Suppress ConstantInputWarning for Static models
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_steam, _ = pearsonr(merged_steam['Model_Ratio'], merged_steam['Positive_Ratio'])
        corr_stock, _ = pearsonr(merged_stock['Model_Ratio'], merged_stock['Stock_Price'])
    
    print("\n" + "="*40)
    print(f"Evaluation Results: [{args.model_name}]")
    print("="*40)
    print(f"1. Correlation (Steam): {corr_steam:.4f}")
    print(f"2. Correlation (Stock): {corr_stock:.4f}")
    if np.isnan(corr_steam):
        print("   (Note: NaN is expected for Static models due to zero variance.)")
    print("="*40 + "\n")

    # Visualization
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(merged_steam['Date'], merged_steam['Positive_Ratio'], 'b-', label='Steam GT', alpha=0.6)
    plt.plot(merged_steam['Date'], merged_steam['Model_Ratio'], 'r--o', label='Prediction', linewidth=2)
    plt.title(f'{args.model_name} vs Steam (r={corr_steam:.2f})')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(merged_stock['Date'], merged_stock['Stock_Price'], 'g-', label='Stock Price', alpha=0.6)
    ax2.plot(merged_stock['Date'], merged_stock['Model_Ratio'], 'r--o', label='Prediction', linewidth=2)
    ax1.set_ylabel('Stock Price')
    ax2.set_ylabel('Purchase Probability')
    plt.title(f'{args.model_name} vs Stock (r={corr_stock:.2f})')
    
    # [수정됨] png 폴더에 저장
    if not os.path.exists("png"):
        os.makedirs("png")
    save_path = f"png/eval_{args.model_name}_graph.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    main()

