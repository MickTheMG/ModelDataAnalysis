from pathlib import Path

from .configs import Config
from .parser import MetricsFactory, ModelTrainingFactory
from .collectors import Collector
from .analyzers import Analyzer
from .visualizers import Visualizer

def main():
    config = Config()
    
    metrics_factory = MetricsFactory(config.column_mapping)
    model_factory = ModelTrainingFactory(metrics_factory)

    collector = Collector(config.input_root, model_factory)
    trainings = collector.collect()
    print(f"Всего было {len(trainings)} экспериментов, найденных в {config.input_root}")

    if not trainings:
        print("Не найдено ни одного эксперимента для анализа.")
        
        config.output_root.mkdir(parents=True, exist_ok=True)
        combined_output_path = config.combined_output_dir
        combined_output_path.mkdir(parents=True, exist_ok=True)
        print(f"Созданы базовые папки: {config.output_root}, {combined_output_path}")
        return

    analyzer = Analyzer(trainings)

    available = analyzer.list_available_metrics()
    print("Доступные метрики:", available)

    if not available:
         print("Не найдено доступных метрик для анализа.")
         return

    metric_of_interest = 'metrics/mAP50-95(B)'  
    if metric_of_interest not in available:
        print(f"Метрика {metric_of_interest} не найдена. Доступны: {available}")
        return
        
  
    best_per_dataset_df = analyzer.find_best_training_per_dataset(metric_of_interest)
    print(f"Лучшие модели по {metric_of_interest}:\n", best_per_dataset_df)

    unique_datasets = analyzer.df['dataset_source'].unique()
    for dataset_name in unique_datasets:
        if dataset_name == 'unknown':
            continue 
        
        dataset_specific_output_dir = config.output_root / f"output_{dataset_name}"
        viz_individual = Visualizer(dataset_specific_output_dir)
        
        df_filtered = analyzer.df[analyzer.df['dataset_source'] == dataset_name]
        if not df_filtered.empty:
            viz_individual.plot_stage_progress(df_filtered, metric_of_interest)
            viz_individual.plot_combined_progress(df_filtered, metric_of_interest)
           
            df_filtered.to_csv(dataset_specific_output_dir / f'all_metrics_{dataset_name}.csv', index=False)


    combined_output_path = config.combined_output_dir
    combined_output_path.mkdir(parents=True, exist_ok=True) 
    
    
    viz_combined = Visualizer(combined_output_path)

    if not best_per_dataset_df.empty:
        best_model_names = best_per_dataset_df['model'].tolist()
        df_best_progress = analyzer.df[analyzer.df['model'].isin(best_model_names)].copy()
        
        if not df_best_progress.empty:
            df_best_progress['model_with_source'] = df_best_progress.apply(lambda row: f"{row['model']} ({row['dataset_source']})", axis=1)
            
            viz_combined.plot_combined_best_progress(df_best_progress, metric_of_interest, combined_output_path)

    best_per_dataset_df.to_csv(combined_output_path / 'best_models_summary.csv', index=False)
    analyzer.df.to_csv(config.output_root / 'all_metrics_full.csv', index=False)

if __name__ == "__main__":
    main()