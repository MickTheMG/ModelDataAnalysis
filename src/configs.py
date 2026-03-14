from pydantic_settings import BaseSettings
from pathlib import Path

class Config(BaseSettings):
    model_root: Path = Path(__file__).parent.parent / "models"
    output_dir: Path = Path(__file__).parent.parent / "outputs"   
    column_mapping: dict = {
        'precision': 'metrics/precision(B)',
        'recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)'
    }