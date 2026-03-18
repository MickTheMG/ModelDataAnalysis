import pandas as pd
from typing import  List, Set, Optional
from .models import ModelTraining

class Analyzer:
    def __init__(self, trainings: List[ModelTraining]):
        self.trainings = trainings
        self.df = self._build_dataframe()
        
    def _build_dataframe(self) -> pd.DataFrame:
        if not self.trainings: 
            return pd.DataFrame(columns=['model', 'epoch_stage', 'path', 'dataset_source'])
        
        records = []
        for t in self.trainings:
            record = {
                'model': t.model_name,
                'epoch_stage': t.epoch_stage,
                'path': str(t.path),
                'dataset_source': getattr(t, 'dataset_source', 'unknown')
            }
            
            if t.all_metrics:
                record.update(t.all_metrics)
            records.append(record)
            
        df = pd.DataFrame(records)
        df = df.sort_values(['model', 'epoch_stage'])
        return df

    def get_summary(self, metric: str) -> pd.DataFrame:
        return self.df.pivot(index='epoch_stage', columns='model', values=metric)

    def compare_models_at_stage(self, epoch_stage: int, metric: Optional[str] = None) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
        
        stage_df = self.df[self.df['epoch_stage'] == epoch_stage].set_index('model')
        if metric:
            return stage_df[[metric]]
        return stage_df

    def list_available_metrics(self) -> List[str]:
        if self.df.empty:
            return []
        
        exclude = {'model', 'epoch_stage', 'path', 'dataset_source'}
        return [col for col in self.df.columns if col not in exclude]
    
    def find_best_training_per_dataset(self, metric: str ) -> pd.DataFrame:
        if self.df.empty or metric not in self.df.columns:
            return pd.DataFrame(columns=['model', 'dataset_source', 'epoch_stage', metric])
        
        index_best = self.df.groupby('dataset_source')[metric].idxmax()
        
        best_trainings_df = self.df.loc[index_best]
        
        return best_trainings_df[['model', 'dataset_source', 'epoch_stage', metric]].reset_index(drop=True)
    
    