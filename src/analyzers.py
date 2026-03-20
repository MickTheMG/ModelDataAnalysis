import pandas as pd
from typing import List
from .models import ModelTraining

class Analyzer:
    def __init__(self, trainings: List[ModelTraining]):
        self.trainings = trainings
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        records = []
        for t in self.trainings:
            record = {
                'model': t.model_name,
                'epoch_stage': t.epoch_stage,
                'path': str(t.path),
                'dataset_source': t.dataset_source
            }
            if t.all_metrics:
                record.update(t.all_metrics)
            records.append(record)
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df = df.sort_values(['dataset_source', 'model', 'epoch_stage']).reset_index(drop=True)
        return df

    def list_available_metrics(self) -> List[str]:
        exclude = {'model', 'epoch_stage', 'path', 'dataset_source'}
        return [col for col in self.df.columns if col not in exclude]

    def get_dataset_progress(self, dataset_source: str, metric: str) -> pd.DataFrame:
        if self.df.empty or metric not in self.df.columns:
            return pd.DataFrame(columns=['model', 'epoch_stage', metric])

        df_dataset = self.df[self.df['dataset_source'] == dataset_source].copy()
        if df_dataset.empty:
            return pd.DataFrame(columns=['model', 'epoch_stage', metric])

        df_dataset[metric] = pd.to_numeric(df_dataset[metric], errors='coerce')
        df_dataset = df_dataset.dropna(subset=[metric])
        if df_dataset.empty:
            return pd.DataFrame(columns=['model', 'epoch_stage', metric])

        return (
            df_dataset[['model', 'epoch_stage', metric]]
            .sort_values(['model', 'epoch_stage'])
            .reset_index(drop=True)
        )

    def find_best_training_per_dataset(self, metric: str) -> pd.DataFrame:
        if self.df.empty or metric not in self.df.columns:
            return pd.DataFrame(columns=['dataset_source', 'model', 'epoch_stage', metric])

        df = self.df.copy()
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric])
        if df.empty:
            return pd.DataFrame(columns=['dataset_source', 'model', 'epoch_stage', metric])

        last_epoch_idx = df.groupby(['dataset_source', 'model'])['epoch_stage'].idxmax()
        last_df = df.loc[last_epoch_idx]

        best_idx = last_df.groupby('dataset_source')[metric].idxmax()
        best_df = last_df.loc[best_idx].reset_index(drop=True)
        best_df = best_df.sort_values('dataset_source').reset_index(drop=True)

        return best_df[['dataset_source', 'model', 'epoch_stage', metric]]

    def get_best_models_progress(self, best_df: pd.DataFrame, metric: str) -> pd.DataFrame:
        if self.df.empty or best_df.empty or metric not in self.df.columns:
            return pd.DataFrame(columns=['dataset_source', 'model', 'epoch_stage', metric, 'model_label'])

        best_keys = best_df[['dataset_source', 'model']].drop_duplicates()
        merged = self.df.merge(best_keys, on=['dataset_source', 'model'], how='inner')
        if merged.empty:
            return pd.DataFrame(columns=['dataset_source', 'model', 'epoch_stage', metric, 'model_label'])

        merged = merged[['dataset_source', 'model', 'epoch_stage', metric]].copy()
        merged[metric] = pd.to_numeric(merged[metric], errors='coerce')
        merged = merged.dropna(subset=[metric])
        if merged.empty:
            return pd.DataFrame(columns=['dataset_source', 'model', 'epoch_stage', metric, 'model_label'])

        merged['model_label'] = merged['model'] + " (" + merged['dataset_source'] + ")"
        return merged.sort_values(['model_label', 'epoch_stage']).reset_index(drop=True)