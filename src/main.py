from .configs import Config
from .parser import MetricsFactory, ModelTrainingFactory
from .collectors import Collector
from .analyzers import Analyzer
from .visualizers import Visualizer
import re

def _normalize_metric_name(metric_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", metric_name.lower())

def _resolve_metric_column(available_metrics, config: Config) -> str:
    requested_metric = config.metric_of_interest
    direct_candidate = config.resolve_metric_name(requested_metric)

    for candidate in (requested_metric, direct_candidate):
        if candidate in available_metrics:
            return candidate

    normalized_requested = _normalize_metric_name(requested_metric)
    for column in available_metrics:
        if _normalize_metric_name(column) == normalized_requested:
            return column

    for short_name, full_name in config.column_mapping.items():
        if _normalize_metric_name(short_name) == normalized_requested and full_name in available_metrics:
            return full_name

    return ""

def _remove_legacy_outputs(config: Config) -> None:
    legacy_files = [config.output_root / "all_metrics_full.csv"]
    for legacy_path in legacy_files:
        if legacy_path.exists():
            legacy_path.unlink()

    dataset_patterns = (
        "all_metrics*.csv",
        "metrics_*_combined.png",
    )
    for dataset_dir in config.output_root.glob("output_*"):
        if not dataset_dir.is_dir():
            continue
        for pattern in dataset_patterns:
            for stale_file in dataset_dir.glob(pattern):
                stale_file.unlink()

    combined_patterns = (
        "metrics_*_best_models_combined_progress.png",
        "metrics_*_all_models_grouped_horizontal_bar.png",
    )
    for pattern in combined_patterns:
        for stale_file in config.combined_output_dir.glob(pattern):
            stale_file.unlink()

def main():
    config = Config()
    metrics_factory = MetricsFactory(config.column_mapping)
    model_factory = ModelTrainingFactory(metrics_factory)

    collector = Collector(
        config.input_root,
        model_factory,
        epoch_step=config.epoch_step,
        result_filenames=config.result_filenames,
    )
    trainings = collector.collect()
    print(f"Найдено {len(trainings)} точек прогресса в {config.input_root}")

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.combined_output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_outputs(config)

    if not trainings:
        print("Не найдено ни одного эксперимента для анализа.")
        return

    analyzer = Analyzer(trainings)
    available = analyzer.list_available_metrics()
    print("Доступные метрики:", available)

    metric_of_interest = _resolve_metric_column(available, config)
    if not metric_of_interest:
        print(
            f"Метрика '{config.metric_of_interest}' не найдена. "
            f"Доступные метрики: {available}"
        )
        return

    safe_metric = Visualizer._safe_metric(metric_of_interest)
    print(f"Выбрана метрика: {metric_of_interest}")

    unique_datasets = sorted(analyzer.df['dataset_source'].dropna().unique())
    for dataset_source in unique_datasets:
        progress_df = analyzer.get_dataset_progress(dataset_source, metric_of_interest)
        if progress_df.empty:
            print(f"[{dataset_source}] Нет данных для метрики '{metric_of_interest}'.")
            continue

        out_dir = config.output_root / f"output_{dataset_source}"
        out_dir.mkdir(parents=True, exist_ok=True)

        output_table = progress_df.rename(columns={'epoch_stage': 'epoch'})
        output_table.to_csv(
            out_dir / f"{safe_metric}_every_{config.epoch_step}_epochs.csv",
            index=False,
        )

        viz = Visualizer(out_dir)
        viz.plot_combined_progress(
            progress_df,
            metric_of_interest,
            file_name=f"{safe_metric}_progress.png",
            title=f"Общий прогресс ({dataset_source})",
        )

    best_per_group = analyzer.find_best_training_per_dataset(metric_of_interest)
    combined_dir = config.combined_output_dir

    if best_per_group.empty:
        print("Не удалось определить лучшие модели по группам.")
        return

    print("Лучшие модели по входным папкам:")
    for _, row in best_per_group.iterrows():
        metric_value = float(row[metric_of_interest])
        print(
            f"- {row['dataset_source']}: {row['model']} "
            f"(эпоха {int(row['epoch_stage'])}) -> {metric_of_interest}={metric_value:.6f}"
        )

    viz_combined = Visualizer(combined_dir)
    best_progress_df = analyzer.get_best_models_progress(best_per_group, metric_of_interest)
    viz_combined.plot_combined_best_progress(best_progress_df, metric_of_interest, combined_dir)
    viz_combined.plot_all_models_grouped_horizontal_bar(
        analyzer.df,
        metric_of_interest,
        file_name=f"{safe_metric}_all_models_grouped_horizontal_bar.png",
    )

    best_per_group = best_per_group.rename(columns={metric_of_interest: 'metric_value'})
    best_per_group['metric_name'] = metric_of_interest
    best_per_group.to_csv(combined_dir / 'best_models_summary.csv', index=False)

    print("Анализ завершён.")

if __name__ == "__main__":
    main()