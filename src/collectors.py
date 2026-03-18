from pathlib import Path
from typing import List
from .parser import ModelTrainingFactory
from .models import ModelTraining

class Collector:
    def __init__(self, root_path: Path, factory: ModelTrainingFactory):
        self.root_path = root_path
        self.factory = factory
        
    def collect(self) -> List[ModelTraining]:
        trainings = []
        
        for stage_dir in self.root_path.rglob('*_e*'):
            if stage_dir.is_dir():
                try:
                    training = self.factory.from_path(stage_dir)
                    
                    training.dataset_source = Path(stage_dir).relative_to(self.root_path).parts[0]
                    trainings.append(training)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Пропущена директория {stage_dir}: {e}")
                    continue
        return trainings
                    
    
    