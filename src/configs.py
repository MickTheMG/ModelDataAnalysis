from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Dict, Tuple

class Config(BaseSettings):
    input_root: Path = Path(__file__).parent.parent / "inputs"
    output_root: Path = Path(__file__).parent.parent / "outputs"
    combined_output_dir_name: str = "combined_results"
    metric_of_interest: str = "mAP50-95"
    epoch_step: int = 50
    result_filenames: Tuple[str, ...] = ("results.csv", "result.csv")

    column_mapping: Dict[str, str] = {
        'precision': 'metrics/precision(B)',
        'recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)'
    }

    @property
    def combined_output_dir(self) -> Path:
        return self.output_root / self.combined_output_dir_name

    def resolve_metric_name(self, metric_name: str) -> str:
        return self.column_mapping.get(metric_name, metric_name)