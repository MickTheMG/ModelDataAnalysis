import pandas as pd
from pathlib import Path
from .models import Metrics, ModelTraining

class MetricsFactory:
    def __init__(self, column_mapping: dict):
        self.mapping = column_mapping
        
    def from_dict(self, data: dict) -> Metrics:
        return Metrics(
            precision=data[self.mapping['precision']],
            recall=data[self.mapping['recall']],
            mAP50=data[self.mapping['mAP50']],
            mAP50_95=data[self.mapping['mAP50-95']]
        )
    
class ModelTrainingFactory:
    def __init__(self, metrics_factory: MetricsFactory):
        self.metrics_factory = metrics_factory

    def from_path(self, path: Path) -> ModelTraining:
        folder_name = path.name
        
        parts = folder_name.split('_e')
        if len(parts) != 2:
            raise ValueError(f"Имя папки {folder_name} не соотв шаблону") 
        
        model_name, stage_str = parts
        stage = int(stage_str)

        csv_path = path / 'results.csv'    
        if not csv_path.exists():
            raise FileNotFoundError(f"Файл {csv_path} не найден")
        
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f".csv файл {csv_path} пустой")
        
        last_row = df.iloc[-1].to_dict()
        final_metrics = self.metrics_factory.from_dict(last_row)

        return ModelTraining(
            model_name=model_name,
            stage=stage,
            path=path,
            final_metrics=final_metrics,
            all_metrics=last_row
        )