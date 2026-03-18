from pydantic_settings import BaseSettings
from pathlib import Path

class Config(BaseSettings):
    
    input_root: Path = Path(__file__).parent.parent / "inputs"
    output_root: Path = Path(__file__).parent.parent / "outputs"   
    combined_output_dir_name: str = "combined_results"
    
    column_mapping: dict = {
        'precision': 'metrics/precision(B)',
        'recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)'
    }
    
    @property
    def combined_output_dir(self) -> Path:
        return self.output_root / self.combined_output_dir_name
    
    