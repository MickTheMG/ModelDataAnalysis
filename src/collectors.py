from pathlib import Path
from typing import List, Optional, Tuple
import re
import pandas as pd
from .parser import ModelTrainingFactory
from .models import ModelTraining

class Collector:
    def __init__(
        self,
        root_path: Path,
        factory: ModelTrainingFactory,
        epoch_step: int = 50,
        result_filenames: Tuple[str, ...] = ("results.csv", "result.csv"),
    ):
        self.root_path = root_path
        self.factory = factory
        self.epoch_step = epoch_step
        self.result_filenames = result_filenames

    def collect(self) -> List[ModelTraining]:
        trainings = []

        if not self.root_path.exists():
            return trainings

        for dataset_dir in sorted(self.root_path.iterdir(), key=lambda p: p.name):
            if not dataset_dir.is_dir():
                continue
            trainings.extend(self._collect_dataset_dir(dataset_dir))

        return trainings

    def _collect_dataset_dir(self, dataset_dir: Path) -> List[ModelTraining]:
        dataset_source = dataset_dir.name
        trainings: List[ModelTraining] = []

        for child_dir in sorted(dataset_dir.iterdir(), key=lambda p: p.name):
            if not child_dir.is_dir():
                continue

            stage_entries = self._find_stage_entries(child_dir)
            if stage_entries:
                trainings.extend(
                    self._collect_manual_stages(
                        dataset_source=dataset_source,
                        model_dir=child_dir,
                        stage_entries=stage_entries,
                    )
                )
                continue

            csv_path = self._find_results_file(child_dir)
            if csv_path is not None:
                trainings.extend(
                    self._collect_single_run(
                        dataset_source=dataset_source,
                        run_dir=child_dir,
                        csv_path=csv_path,
                    )
                )

        return trainings

    def _collect_manual_stages(
        self,
        dataset_source: str,
        model_dir: Path,
        stage_entries: List[Tuple[int, Path, Path]],
    ) -> List[ModelTraining]:
        trainings: List[ModelTraining] = []
        model_name = self._extract_model_name(model_dir.name)

        for epoch_stage, stage_dir, csv_path in stage_entries:
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                row = df.iloc[-1].to_dict()
                metrics = self.factory.metrics_factory.from_dict(row)
                trainings.append(
                    ModelTraining(
                        model_name=model_name,
                        epoch_stage=epoch_stage,
                        path=stage_dir,
                        final_metrics=metrics,
                        all_metrics=row,
                        dataset_source=dataset_source,
                    )
                )
            except Exception as exc:
                print(f"Пропущена папка {stage_dir}: {exc}")

        return trainings

    def _collect_single_run(
        self,
        dataset_source: str,
        run_dir: Path,
        csv_path: Path,
    ) -> List[ModelTraining]:
        trainings: List[ModelTraining] = []

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return trainings

            if "epoch" in df.columns:
                epoch_series = pd.to_numeric(df["epoch"], errors="coerce")
            else:
                epoch_series = pd.Series(range(1, len(df) + 1), index=df.index)

            df = df.copy()
            df["_epoch_stage"] = epoch_series
            df = df.dropna(subset=["_epoch_stage"])
            if df.empty:
                return trainings

            df["_epoch_stage"] = df["_epoch_stage"].astype(int)
            selected_df = df[df["_epoch_stage"] % self.epoch_step == 0].sort_values("_epoch_stage")
            if selected_df.empty:
                return trainings

            model_name = self._extract_model_name(run_dir.name)
            for _, row in selected_df.iterrows():
                row_dict = row.to_dict()
                epoch_stage = int(row_dict.pop("_epoch_stage"))
                metrics = self.factory.metrics_factory.from_dict(row_dict)
                trainings.append(
                    ModelTraining(
                        model_name=model_name,
                        epoch_stage=epoch_stage,
                        path=run_dir,
                        final_metrics=metrics,
                        all_metrics=row_dict,
                        dataset_source=dataset_source,
                    )
                )

        except Exception as exc:
            print(f"Ошибка при обработке {run_dir}: {exc}")

        return trainings

    def _find_stage_entries(self, model_dir: Path) -> List[Tuple[int, Path, Path]]:
        stage_entries: List[Tuple[int, Path, Path]] = []

        for stage_dir in model_dir.iterdir():
            if not stage_dir.is_dir():
                continue

            epoch_stage = self._extract_epoch(stage_dir.name)
            if epoch_stage is None:
                continue
            if epoch_stage % self.epoch_step != 0:
                continue

            csv_path = self._find_results_file(stage_dir)
            if csv_path is None:
                continue

            stage_entries.append((epoch_stage, stage_dir, csv_path))

        stage_entries.sort(key=lambda entry: entry[0])
        return stage_entries

    def _find_results_file(self, directory: Path) -> Optional[Path]:
        for file_name in self.result_filenames:
            csv_path = directory / file_name
            if csv_path.exists():
                return csv_path
        return None

    @staticmethod
    def _extract_epoch(folder_name: str) -> Optional[int]:
        match = re.search(r"_e(\d+)", folder_name)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_model_name(raw_name: str) -> str:
        prefix_before_epoch = re.split(r"_e\d+.*", raw_name, maxsplit=1)[0]
        if prefix_before_epoch:
            cleaned = re.sub(r"_b\d+$", "", prefix_before_epoch)
            if cleaned:
                return cleaned

        match = re.search(r"(yolo[^_/\\]*)", raw_name, flags=re.IGNORECASE)
        if match:
            return match.group(1)

        return raw_name