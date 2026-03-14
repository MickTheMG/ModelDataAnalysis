import pandas as pd
from typing import  List, Set, Optional
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
                'stage': t.stage,
                'path': str(t.path)
            }
            
            if t.all_metrics:
                record.update(t.all_metrics)
            records.append(record)
        df = pd.DataFrame(records)
        df = df.sort_values(['model', 'stage'])
        return df

    def get_summary(self, metric: str) -> pd.DataFrame:
        return self.df.pivot(index='stage', columns='model', values=metric)

    def compare_models_at_stage(self, stage: int, metric: Optional[str] = None) -> pd.DataFrame:
        stage_df = self.df[self.df['stage'] == stage].set_index('model')
        if metric:
            return stage_df[[metric]]
        return stage_df

    def list_available_metrics(self) -> List[str]:
        exclude = {'model', 'stage', 'path'}
        return [col for col in self.df.columns if col not in exclude]