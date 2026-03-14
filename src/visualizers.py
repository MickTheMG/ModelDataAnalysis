from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_model_comparison(self, df: pd.DataFrame, metric: str, title: str = None):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='stage', y=metric, hue='model')
        
        if title is None:
            title = f'Сравнение моделей по {metric}'
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(self.output_dir / f'{safe_metric}_comparison.png')
        plt.close()

    def plot_stage_progress(self, df: pd.DataFrame, metric: str):
        g = sns.FacetGrid(df, col='model', hue='model', col_wrap=3, sharey=False)
        g.map(sns.lineplot, 'stage', metric, marker='o')
        g.add_legend()
        
        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(self.output_dir / f'{safe_metric}_progress.png')
        plt.close()

    def plot_heatmap(self, df: pd.DataFrame, metric: str):
        pivot = df.pivot(index='model', columns='stage', values=metric)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'Тепловая карта {metric}')
        
        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(self.output_dir / f'{safe_metric}_heatmap.png')
        plt.close()