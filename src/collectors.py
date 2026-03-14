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
        
        for model_dir in self.root_path.iterdir():
            if not model_dir.is_dir():
                continue
            for stage_dir in model_dir.glob('*_e*'):
                if stage_dir.is_dir():
                    training = self.factory.from_path(stage_dir)
                    trainings.append(training)
        return trainings
    
    