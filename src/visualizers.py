from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from decimal import Decimal, ROUND_HALF_UP
import re

class Visualizer:
    TRAINING_GROUPS: Tuple[Tuple[str, str], ...] = (
        ("single_dataset", "Training of all models on 1 dataset"),
        ("fusion_two_datasets_one_filter", "Fusion of 2 datasets with one filter"),
        ("fusion_two_datasets_different_filters", "Fusion of 2 datasets with different filters"),
        ("all_datasets", "Training all models on all datasets"),
    )
    FIGSIZE_A4: Tuple[float, float] = (8.72, 11.69)
    FIGSIZE_DEFAULT: Tuple[float, float] = (10, 6)
    BLOCK_VERTICAL_GAP: float = 0.35
    DATASET_BLOCK_GAP: float = 0.55
    MODEL_ROW_STEP: float = 1.25
    HORIZONTAL_BAR_HEIGHT: float = 0.65
    FONT_FAMILY: str = 'serif'
    FONT_SERIF: Tuple[str, str] = ('Times New Roman', 'DejaVu Serif')
    FONT_TITLE: int = 11
    FONT_TICKS: int = 9
    FONT_AXIS: int = 10
    FONT_ANNOTATION: int = 8
    FONT_SUPTITLE: int = 13
    MODEL_FAMILY_BASE_COLORS: Dict[str, Tuple[float, float, float]] = {
        'yolo8': (0.6471, 0.3490, 0.6667),   # purple  (#a559aa)
        'yolo11': (0.3490, 0.6588, 0.6118),  # teal    (#59a89c)
        'yolo12': (0.9412, 0.7725, 0.4431),  # gold    (#f0c571)
        'yolo26': (0.8078, 0.8078, 0.8078),  # gray    (#cecece)
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _apply_plot_style(cls) -> None:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': cls.FONT_FAMILY,
            'font.serif': list(cls.FONT_SERIF),
        })

    @staticmethod
    def _safe_metric(metric: str) -> str:
        return metric.replace('/', '_').replace('(', '').replace(')', '')

    @staticmethod
    def _normalize_dataset_source(dataset_source: str) -> str:
        return dataset_source.lower().replace('-', '_').replace(' ', '_')

    @classmethod
    def _resolve_training_group(cls, dataset_source: str) -> str:
        normalized = cls._normalize_dataset_source(dataset_source)

        all_dataset_tokens = (
            "all_datasets",
            "all_dataset",
            "all_data",
            "all_data_sources",
            "all_sources",
        )
        if any(token in normalized for token in all_dataset_tokens):
            return "all_datasets"

        different_filter_direct_tokens = ("l1_l2",)
        if any(token in normalized for token in different_filter_direct_tokens):
            return "fusion_two_datasets_different_filters"

        two_dataset_tokens = ("2_datasets", "2_dataset", "two_datasets", "two_dataset")
        fusion_tokens = ("fusion", "merge", "concat", "joined", "sraschen")
        is_two_dataset_fusion = any(token in normalized for token in two_dataset_tokens) or (
            "2" in normalized and any(token in normalized for token in fusion_tokens)
        )

        if is_two_dataset_fusion:
            different_filter_tokens = (
                "different_filters",
                "different_filter",
                "diff_filters",
                "diff_filter",
                "multi_filter",
                "raznye_filtry",
                "raznymi_filtrami",
            )
            if any(token in normalized for token in different_filter_tokens):
                return "fusion_two_datasets_different_filters"

            same_filter_tokens = (
                "same_filter",
                "one_filter",
                "single_filter",
                "odin_filter",
                "odnim_filtrom",
            )
            if any(token in normalized for token in same_filter_tokens):
                return "fusion_two_datasets_one_filter"

            # По умолчанию fusion_2_datasets относим к варианту "с одним фильтром".
            return "fusion_two_datasets_one_filter"

        return "single_dataset"

    @staticmethod
    def _model_sort_key(model_name: str) -> Tuple[int, int, str]:
        normalized = model_name.lower()
        family_match = re.search(r"yolov?(\d+)", normalized)
        family = int(family_match.group(1)) if family_match else 10**6

        size_order = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}
        size_match = re.search(r"yolov?\d*([nsmlx])(?:\b|_)", normalized)
        size_value = size_order.get(size_match.group(1), 99) if size_match else 99

        return family, size_value, normalized

    @staticmethod
    def _resolve_model_family(model_name: str) -> str:
        normalized = model_name.lower()
        family_match = re.search(r"yolov?(\d+)", normalized)
        if family_match:
            return f"yolo{family_match.group(1)}"
        return "other"

    @staticmethod
    def _short_model_name(model_name: str) -> str:
        return re.sub(r"(?i)^yolov?_?", "", model_name).strip()

    @classmethod
    def _build_family_color_map(cls, families: List[str]) -> Dict[str, Tuple[float, float, float]]:
        color_map: Dict[str, Tuple[float, float, float]] = {}
        for family in families:
            if family in cls.MODEL_FAMILY_BASE_COLORS:
                color_map[family] = cls.MODEL_FAMILY_BASE_COLORS[family]

        unknown_families = [family for family in families if family not in color_map]
        if unknown_families:
            fallback_palette = sns.color_palette('tab20', n_colors=len(unknown_families))
            for family, color in zip(unknown_families, fallback_palette):
                color_map[family] = color

        return color_map

    @staticmethod
    def _dataset_source_sort_key(dataset_source: str) -> Tuple[int, str]:
        normalized = dataset_source.lower()
        family_match = re.search(r"(?:models?|model)_?(\d+)", normalized)
        family = int(family_match.group(1)) if family_match else 10**6
        return family, normalized

    @staticmethod
    def _darken_color(color: Tuple[float, float, float], amount: float = 0.28) -> Tuple[float, float, float]:
        amount = max(0.0, min(1.0, amount))
        return tuple(max(0.0, min(1.0, channel * (1.0 - amount))) for channel in color)

    @staticmethod
    def _lighten_color(color: Tuple[float, float, float], amount: float = 0.35) -> Tuple[float, float, float]:
        amount = max(0.0, min(1.0, amount))
        return tuple(max(0.0, min(1.0, channel + (1.0 - channel) * amount)) for channel in color)

    @staticmethod
    def _blend_colors(
        start_color: Tuple[float, float, float],
        end_color: Tuple[float, float, float],
        ratio: float,
    ) -> Tuple[float, float, float]:
        ratio = max(0.0, min(1.0, ratio))
        return tuple(
            max(0.0, min(1.0, start + (end - start) * ratio))
            for start, end in zip(start_color, end_color)
        )

    @classmethod
    def _gradient_color_for_model(
        cls,
        model_name: str,
        model_color_map: Dict[str, Tuple[float, float, float]],
        model_seen_count: Dict[str, int],
        model_total_count: Dict[str, int],
    ) -> Tuple[float, float, float]:
        base_color = model_color_map[model_name]
        dark_color = cls._darken_color(base_color)
        light_color = cls._lighten_color(base_color)

        seen_idx = model_seen_count.get(model_name, 0)
        total = model_total_count.get(model_name, 1)
        model_seen_count[model_name] = seen_idx + 1

        ratio = 0.4 if total <= 1 else seen_idx / float(total - 1)
        return cls._blend_colors(dark_color, light_color, ratio)

    @staticmethod
    def _format_metric_value(value: float) -> str:
        rounded = Decimal(str(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return f"{rounded:.2f}"

    def _build_horizontal_bar_positions(self, dataset_sources: pd.Series) -> List[float]:
        y_positions: List[float] = []
        current_y = 0.0
        previous_source = None

        for source_name in dataset_sources:
            if previous_source is not None:
                current_y += self.MODEL_ROW_STEP
                if source_name != previous_source:
                    current_y += self.DATASET_BLOCK_GAP
            y_positions.append(current_y)
            previous_source = source_name

        return y_positions

    def _plot_horizontal_bar_panel(
        self,
        ax,
        group_title: str,
        panel_label: str,
        group_df: pd.DataFrame,
        metric: str,
        left_limit: float,
        right_limit: float,
        family_color_map: Dict[str, Tuple[float, float, float]],
        x_tick_step: float,
    ) -> None:
        ax.grid(axis='x', color='#d9d9d9', linestyle='--', linewidth=0.7, alpha=0.8)
        ax.grid(axis='y', visible=False)
        ax.set_ylabel('')
        ax.set_xlim(left_limit, right_limit)
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
        ax.tick_params(axis='x', labelsize=self.FONT_AXIS)
        ax.tick_params(axis='y', labelsize=self.FONT_TICKS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        panel_title = f"{panel_label} {group_title}"
        if group_df.empty:
            ax.set_title(f'{panel_title} (no data)', loc='left', fontsize=self.FONT_TITLE, fontweight='bold')
            ax.set_yticks([])
            return

        group_df = group_df.reset_index(drop=True)
        colors = [family_color_map[family_name] for family_name in group_df['model_family']]
        y_positions = self._build_horizontal_bar_positions(group_df['dataset_source'])

        bars = ax.barh(
            y_positions,
            group_df[metric],
            color=colors,
            edgecolor='#5f5f5f',
            linewidth=0.8,
            height=self.HORIZONTAL_BAR_HEIGHT,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(group_df['display_label'])
        ax.invert_yaxis()
        ax.set_title(panel_title, loc='left', fontsize=self.FONT_TITLE, fontweight='bold')

        text_shift = max(0.004, (right_limit - left_limit) * 0.008)
        for bar, value in zip(bars, group_df[metric]):
            text_x = min(float(value) + text_shift, right_limit - text_shift)
            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                self._format_metric_value(float(value)),
                va='center',
                ha='left',
                fontsize=self.FONT_ANNOTATION,
            )

    @staticmethod
    def _remove_right_third_of_axis(ax) -> None:
        position = ax.get_position()
        ax.set_position([position.x0, position.y0, position.width * (2.0 / 3.0), position.height])

    def plot_combined_progress(
        self,
        df: pd.DataFrame,
        metric: str,
        file_name: str = None,
        title: str = None,
        hue_column: str = "model",
    ):
        if df.empty:
            return

        self._apply_plot_style()
        plt.figure(figsize=self.FIGSIZE_DEFAULT)
        sns.lineplot(data=df, x='epoch_stage', y=metric, hue=hue_column, marker='o', palette='tab10')
        plt.title(title or f'Общий прогресс ({metric})')
        plt.xlabel('Эпоха')
        plt.ylabel(metric)
        plt.xticks(sorted(df['epoch_stage'].unique()))
        plt.grid(alpha=0.25)
        plt.legend(loc='upper left', title='Модель', frameon=True)
        plt.tight_layout()

        safe_metric = self._safe_metric(metric)
        output_name = file_name or f'{safe_metric}_combined.png'
        plt.savefig(self.output_dir / output_name)
        plt.close()

    def plot_combined_best_progress(self, df: pd.DataFrame, metric: str, output_path: Path):
        if df.empty:
            return

        self._apply_plot_style()
        plt.figure(figsize=self.FIGSIZE_DEFAULT)
        sns.lineplot(
            data=df,
            x='epoch_stage',
            y=metric,
            hue='model_label',
            marker='o',
            linewidth=2,
            palette='tab10',
        )
        plt.title(f'Прогресс Лучших Моделей по Группам ({metric})')
        plt.xlabel('Эпоха')
        plt.ylabel(metric)
        plt.xticks(sorted(df['epoch_stage'].unique()))
        plt.grid(alpha=0.25)
        plt.legend(loc='upper left', title='Лучшая модель (группа)', frameon=True)
        plt.tight_layout()

        safe_metric = self._safe_metric(metric)
        plt.savefig(output_path / f'{safe_metric}_best_models_progress.png')
        plt.close()

    def plot_all_models_grouped_horizontal_bar(
        self,
        df: pd.DataFrame,
        metric: str,
        file_name: str = None,
        title: str = None,
        x_tick_step: float = 0.1,
    ):
        required_columns = {'dataset_source', 'model', 'epoch_stage', metric}
        if df.empty or not required_columns.issubset(df.columns):
            return

        plot_df = df[['dataset_source', 'model', 'epoch_stage', metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors='coerce')
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            return

        latest_idx = plot_df.groupby(['dataset_source', 'model'])['epoch_stage'].idxmax()
        plot_df = plot_df.loc[latest_idx].copy()
        plot_df['training_group'] = plot_df['dataset_source'].apply(self._resolve_training_group)
        plot_df['model_family'] = plot_df['model'].apply(self._resolve_model_family)

        unique_families = sorted(plot_df['model_family'].unique())
        family_color_map = self._build_family_color_map(unique_families)

        grouped_dfs: List[pd.DataFrame] = []
        for group_key, _ in self.TRAINING_GROUPS:
            group_df = plot_df[plot_df['training_group'] == group_key].copy()
            group_df['dataset_sort_key'] = group_df['dataset_source'].apply(self._dataset_source_sort_key)
            group_df['model_sort_key'] = group_df['model'].apply(self._model_sort_key)
            group_df['short_model'] = group_df['model'].apply(self._short_model_name)
            group_df = group_df.sort_values(['dataset_sort_key', 'model_sort_key']).drop(
                columns=['dataset_sort_key', 'model_sort_key']
            )

            has_duplicates = group_df['model'].duplicated(keep=False).any()
            if has_duplicates:
                group_df['display_label'] = group_df['short_model'] + " (" + group_df['dataset_source'] + ")"
            else:
                group_df['display_label'] = group_df['short_model']

            grouped_dfs.append(group_df)

        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        panel_order = [0, 2, 1, 3]
        ordered_groups = [self.TRAINING_GROUPS[i] for i in panel_order]
        ordered_grouped_dfs = [grouped_dfs[i] for i in panel_order]
        self._apply_plot_style()
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=self.FIGSIZE_A4,
            sharex=False,
            constrained_layout=False,
        )
        axes_flat = axes.flatten()

        if x_tick_step <= 0:
            raise ValueError("x_tick_step must be > 0")

        # Top-left panel
        top_left_left_limit = 0.6
        top_left_right_limit = (
            max(0.55, float(ordered_grouped_dfs[0][metric].max()) + 0.04)
            if not ordered_grouped_dfs[0].empty else 0.55
        )
        self._plot_horizontal_bar_panel(
            ax=axes_flat[0],
            group_title=ordered_groups[0][1],
            panel_label=panel_labels[0],
            group_df=ordered_grouped_dfs[0],
            metric=metric,
            left_limit=top_left_left_limit,
            right_limit=top_left_right_limit,
            family_color_map=family_color_map,
            x_tick_step=x_tick_step,
        )

        # Top-right panel
        top_right_group_df = ordered_grouped_dfs[2]
        top_right_left_limit = 0.6
        top_right_right_limit = (
            max(0.55, float(top_right_group_df[metric].max()) + 0.04)
            if not top_right_group_df.empty else 0.55
        )
        self._plot_horizontal_bar_panel(
            ax=axes_flat[1],
            group_title=ordered_groups[1][1],
            panel_label=panel_labels[1],
            group_df=top_right_group_df,
            metric=metric,
            left_limit=top_right_left_limit,
            right_limit=top_right_right_limit,
            family_color_map=family_color_map,
            x_tick_step=x_tick_step,
        )

        # Bottom-left panel
        bottom_left_group_df = ordered_grouped_dfs[1]
        bottom_left_left_limit = 0.6
        bottom_left_right_limit = (
            max(0.55, float(bottom_left_group_df[metric].max()) + 0.04)
            if not bottom_left_group_df.empty else 0.55
        )
        self._plot_horizontal_bar_panel(
            ax=axes_flat[2],
            group_title=ordered_groups[2][1],
            panel_label=panel_labels[2],
            group_df=bottom_left_group_df,
            metric=metric,
            left_limit=bottom_left_left_limit,
            right_limit=bottom_left_right_limit,
            family_color_map=family_color_map,
            x_tick_step=x_tick_step,
        )

        # Bottom-right panel
        bottom_right_left_limit = 0.6
        bottom_right_right_limit = (
            max(0.55, float(ordered_grouped_dfs[3][metric].max()) + 0.04)
            if not ordered_grouped_dfs[3].empty else 0.55
        )
        self._plot_horizontal_bar_panel(
            ax=axes_flat[3],
            group_title=ordered_groups[3][1],
            panel_label=panel_labels[3],
            group_df=ordered_grouped_dfs[3],
            metric=metric,
            left_limit=bottom_right_left_limit,
            right_limit=bottom_right_right_limit,
            family_color_map=family_color_map,
            x_tick_step=x_tick_step,
        )

        # Show X-axis scale labels and axis titles on top row as well.
        axes_flat[0].tick_params(axis='x', labelbottom=True)
        axes_flat[1].tick_params(axis='x', labelbottom=True)
        axes_flat[0].set_xlabel(metric, fontsize=self.FONT_AXIS)
        axes_flat[1].set_xlabel(metric, fontsize=self.FONT_AXIS)
        axes_flat[2].set_xlabel(metric, fontsize=self.FONT_AXIS)
        axes_flat[3].set_xlabel(metric, fontsize=self.FONT_AXIS)

        fig.suptitle(
            title or f'Comparison of models by learning strategies ({metric})',
            fontsize=self.FONT_SUPTITLE,
            fontweight='bold',
        )
        fig.subplots_adjust(
            left=0.12,
            right=0.97,
            top=0.94,
            bottom=0.05,
            hspace=0.40,
            wspace=0.35,
        )
        safe_metric = self._safe_metric(metric)
        output_name = file_name or f'{safe_metric}_all_models_grouped_horizontal_bar.png'
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()