from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_metric(metric: str) -> str:
        return metric.replace('/', '_').replace('(', '').replace(')', '')

    def plot_combined_progress(
        self,
        df: pd.DataFrame,
        metric: str,
        file_name: str = None,
        title: str = None,
        hue_column: str = "model",
    ):
        if df.empty:
            return

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='epoch_stage', y=metric, hue=hue_column, marker='o', palette='tab10')
        plt.title(title or f'Общий прогресс ({metric})')
        plt.xlabel('Эпоха')
        plt.ylabel(metric)
        plt.xticks(sorted(df['epoch_stage'].unique()))
        plt.grid(alpha=0.25)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Модель')
        plt.tight_layout()

        safe_metric = self._safe_metric(metric)
        output_name = file_name or f'{safe_metric}_combined.png'
        plt.savefig(self.output_dir / output_name)
        plt.close()

    def plot_combined_best_progress(self, df: pd.DataFrame, metric: str, output_path: Path):
        if df.empty:
            return

        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=df,
            x='epoch_stage',
            y=metric,
            hue='model_label',
            marker='o',
            linewidth=2,
            palette='tab10',
        )
        plt.title(f'Прогресс Лучших Моделей по Группам ({metric})')
        plt.xlabel('Эпоха')
        plt.ylabel(metric)
        plt.xticks(sorted(df['epoch_stage'].unique()))
        plt.grid(alpha=0.25)
        plt.legend(title='Лучшая модель (группа)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        safe_metric = self._safe_metric(metric)
        plt.savefig(output_path / f'{safe_metric}_best_models_progress.png')
        plt.close()