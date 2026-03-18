from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_stage_progress(self, df: pd.DataFrame, metric: str):
        g = sns.FacetGrid(df, col='model', hue='model', col_wrap=3, sharey=False)
        g.map(sns.lineplot, 'epoch_stage', metric, marker='o')
        g.add_legend()
        
        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(self.output_dir / f'{safe_metric}_progress.png')
        plt.close()

    def plot_combined_progress(self, df: pd.DataFrame, metric: str):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='epoch_stage', y=metric, hue='model', marker='o')
        plt.title(f'Общий прогресс')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(self.output_dir / f'{safe_metric}_combined.png')
        plt.close()
        
    def plot_model_comparison(self, df: pd.DataFrame, metric: str, title: str = None):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='epoch_stage', y=metric, hue='model')
        
        if title is None:
            title = f'Сравнение по {metric}'
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(self.output_dir / f'{safe_metric}_comparison.png')
        plt.close()
        
    def plot_combined_best_progress(self, df: pd.DataFrame, metric: str, output_path: Path):
        plt.figure(figsize=(12, 8))
        
        sns.lineplot(data=df, x='epoch_stage', y=metric, hue='dataset_source', style='dataset_source', marker='o', markersize=8)
        plt.title(f'Прогресс Лучших Моделей по Наборам Данных ({metric})')
        
        plt.xlabel('Эпоха')
        plt.ylabel(metric)
        
        plt.legend(title='Модель', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        safe_metric = metric.replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(output_path / f'{safe_metric}_best_models_combined_progress.png')
        plt.close()