import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    """
    Generates plots for data leakage auditing.
    """
    
    def __init__(self, output_dir: str = 'reports/figures'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def plot_feature_distribution(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature: str):
        """
        Overlays feature distribution for train and test.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_df[feature], label='Train', fill=True, alpha=0.3)
        sns.kdeplot(test_df[feature], label='Test', fill=True, alpha=0.3)
        plt.title(f'Distribution Shift: {feature}')
        plt.legend()
        plt.savefig(f'{self.output_dir}/dist_{feature}.png')
        plt.close()
        
    def plot_correlation_matrix(self, df: pd.DataFrame, title: str):
        """
        Plots correlation matrix.
        """
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        filename = title.lower().replace(' ', '_')
        plt.savefig(f'{self.output_dir}/{filename}.png')
        plt.close()

    def plot_risk_summary(self, risk_df: pd.DataFrame):
        """
        Bar chart of top risk features.
        """
        if risk_df.empty:
            return
            
        plt.figure(figsize=(10, 6))
        top_risks = risk_df.head(10)
        sns.barplot(x='risk_score', y='feature', data=top_risks, hue='feature', palette='Reds_r')
        plt.title('Top Leaky Features by Risk Score')
        plt.xlim(0, 1.1)
        plt.savefig(f'{self.output_dir}/risk_summary.png')
        plt.close()
