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
    
    collector = Collector(config.model_root, model_factory)
    trainings = collector.collect()
    print(f"Собрано {len(trainings)} тренировок")
    
    analyzer = Analyzer(trainings)
    
    available = analyzer.list_available_metrics()
    print("Доступные метрики:", available)
    
    metric_of_interest = 'metrics/mAP50-95(B)'  
    summary = analyzer.get_summary(metric=metric_of_interest)
    print(f"Сводная таблица ({metric_of_interest}):\n", summary)
    
    viz = Visualizer(config.output_dir)
    viz.plot_model_comparison(analyzer.df, metric_of_interest, f'Сравнение по {metric_of_interest}')
    viz.plot_stage_progress(analyzer.df, metric_of_interest)
    # viz.plot_heatmap(analyzer.df, metric_of_interest)

    analyzer.df.to_csv(config.output_dir / 'all_metrics.csv', index=False)
    
if __name__ == "__main__":
    main()