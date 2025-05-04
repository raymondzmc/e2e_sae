from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from e2e_sae.log import logger


def plot_per_layer_metric(
    df: pd.DataFrame,
    run_ids: Mapping[int, Mapping[str, str]],
    metric: str,
    final_layer: int,
    run_types: Sequence[str],
    out_file: str | Path | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    legend_label_cols_and_precision: list[tuple[str, int]] | None = None,
    legend_title: str | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    horz_layout: bool = False,
    show_ax_titles: bool = True,
    save_svg: bool = True,
) -> None:
    """
    Plot the per-layer metric (explained variance or reconstruction loss) for different run types.

    Args:
        df: DataFrame containing the filtered data for the specific layer.
        run_ids: The run IDs to use. Format: {layer: {run_type: run_id}}.
        metric: The metric to plot ('explained_var' or 'recon_loss').
        final_layer: The final layer to plot up to.
        run_types: The run types to include in the plot.
        out_file: The filename which the plot will be saved as.
        ylim: The y-axis limits.
        legend_label_cols_and_precision: Columns in df that should be used for the legend, along
            with their precision. Added in addition to the run type.
        legend_title: The title of the legend.
        styles: The styles to use.
        horz_layout: Whether to use a horizontal layout for the subplots. Requires sae_layers to be
            exactly [2, 6, 10]. Ignores legend_label_cols_and_precision if True.
        show_ax_titles: Whether to show titles for each subplot.
        save_svg: Whether to save the plot as an SVG file in addition to PNG. Default is True.
    """
    metric_names = {
        "explained_var": "Explained Variance",
        "explained_var_ln": "Explained Variance\nof Normalized Activations",
        "recon_loss": "Reconstruction MSE",
    }
    metric_name = metric_names.get(metric, metric)

    sae_layers = list(run_ids.keys())
    n_sae_layers = len(sae_layers)

    if horz_layout:
        assert sae_layers == [2, 6, 10]
        fig, axs = plt.subplots(
            1, n_sae_layers, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2, 1.2]}
        )
        legend_label_cols_and_precision = None
    else:
        fig, axs = plt.subplots(n_sae_layers, 1, figsize=(5, 3.5 * n_sae_layers))
    axs = np.atleast_1d(axs)  # type: ignore

    def plot_metric(
        ax: plt.Axes,
        plot_df: pd.DataFrame,
        sae_layer: int,
        xs: NDArray[np.signedinteger[Any]],
    ) -> None:
        for _, row in plot_df.iterrows():
            run_type = row["run_type"]
            assert isinstance(run_type, str)
            legend_label = styles[run_type]["label"] if styles is not None else run_type
            if legend_label_cols_and_precision is not None:
                assert all(
                    col in row for col, _ in legend_label_cols_and_precision
                ), f"Legend label cols not found in row: {row}"
                metric_strings = [
                    f"{col}={format(row[col], f'.{prec}f')}"
                    for col, prec in legend_label_cols_and_precision
                ]
                legend_label += f" ({', '.join(metric_strings)})"
            ys = [row[f"{metric}_layer-{i}"] for i in range(sae_layer, final_layer + 1)]
            kwargs = styles[run_type] if styles is not None else {}
            ax.plot(xs, ys, **kwargs)

    for i, sae_layer in enumerate(sae_layers):
        layer_df = df.loc[df["id"].isin(list(run_ids[sae_layer].values()))]

        ax = axs[i]

        xs = np.arange(sae_layer, final_layer + 1)
        for run_type in run_types:
            plot_metric(ax, layer_df.loc[layer_df["run_type"] == run_type], sae_layer, xs)

        if show_ax_titles:
            ax.set_title(f"SAE Layer {sae_layer}", fontweight="bold")
        ax.set_xlabel("Model Layer")
        if (not horz_layout) or i == 0:
            ax.legend(title=legend_title, loc="best")
            ax.set_ylabel(metric_name)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_ylim(ylim)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved to {out_file}")
        if save_svg:
            plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close()


def plot_facet(
    df: pd.DataFrame,
    xs: Sequence[str],
    y: str,
    facet_by: str,
    line_by: str,
    line_by_vals: Sequence[str] | None = None,
    sort_by: str | None = None,
    xlabels: Sequence[str | None] | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    facet_vals: Sequence[Any] | None = None,
    xlims: Sequence[Mapping[Any, tuple[float | None, float | None]] | None] | None = None,
    xticks: Sequence[tuple[list[float], list[str]] | None] | None = None,
    yticks: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[Any, tuple[float | None, float | None]] | None = None,
    styles: Mapping[Any, Mapping[str, Any]] | None = None,
    title: Mapping[Any, str] | None = None,
    legend_title: str | None = None,
    axis_formatter: Callable[[Sequence[plt.Axes]], None] | None = None,
    out_file: str | Path | None = None,
    plot_type: Literal["line", "scatter"] = "line",
    annotate_col: str | None = None,
    save_svg: bool = True,
) -> None:
    """Generates faceted plots with multiple x-axes and a shared y-axis.

    This function creates a figure with potentially multiple rows of subplots (facets).
    Each row corresponds to a unique value in the `facet_by` column of the input DataFrame.
    Within each row, there are multiple subplots arranged horizontally, one for each
    variable specified in the `xs` list.

    Data is grouped by the `line_by` column, and a separate line or set of scatter points
    is drawn for each group (`line_val`) within each subplot. Colors are assigned
    consistently to each `line_val` across all facets using the 'tab10' palette.

    Args:
        df: DataFrame containing the data to plot.
        xs: Sequence of column names in `df` to use as the independent variables
            for the different x-axes within each facet.
        y: Column name in `df` to use as the dependent variable for the shared y-axis.
        facet_by: Column name in `df` to group data into different rows (facets) of subplots.
        line_by: Column name in `df` to group data into separate lines or scatter series
            within each subplot.
        line_by_vals: Optional sequence of specific values from the `line_by` column to plot.
            If None, all unique values in `line_by` are plotted.
        sort_by: Optional column name in `df` used to sort the data points before drawing lines.
            If None and `plot_type` is 'line', defaults to the `y` column.
            Crucial for ensuring lines connect points in the intended order.
        xlabels: Optional sequence of strings to use as labels for the x-axes, corresponding
            to the columns in `xs`. If None, the column names from `xs` are used.
        ylabel: Optional string to use as the label for the y-axis. If None, the column name
            from `y` is used.
        suptitle: Optional string to set as the main title for the entire figure.
        facet_vals: Optional sequence of values from the `facet_by` column. Determines which
            facets are plotted and their order. If None, all unique values are plotted in
            sorted order.
        xlims: Optional sequence, one element per x-axis (matching `xs`). Each element
            can be None or a mapping from `facet_val` to a tuple `(min, max)` defining
            the x-axis limits for that specific axis within that specific facet.
        xticks: Optional sequence, one element per x-axis (matching `xs`). Each element
            can be None or a tuple `(ticks, labels)` for setting custom x-axis tick
            positions and labels for that specific axis across all facets.
        yticks: Optional tuple `(ticks, labels)` for setting custom y-axis tick positions and
            labels. Applied only to the y-axis of the first subplot in each row.
        ylim: Optional mapping from `facet_val` to a tuple `(min, max)` defining the y-axis
            limits for all subplots within that specific facet.
        styles: Optional mapping from `line_val` to a dictionary of matplotlib keyword arguments
            (e.g., {'marker': '*', 'linestyle': '--', 'label': 'Custom Name'})
            to customize the appearance of specific lines/scatter series.
        title: Optional mapping from `facet_val` to a string to be used as the title for
            that specific facet row. If None, titles default to f"{facet_by}={facet_val}".
        legend_title: Optional string to use as the title for the figure legend. If None,
            the column name from `line_by` is used.
        axis_formatter: Optional callable that accepts a sequence of `plt.Axes` (the axes
            for a single facet row) and applies custom formatting.
        out_file: Optional path (string or Path object) where the plot will be saved.
            If None, the plot is not saved.
        plot_type: Specifies the type of plot: 'line' or 'scatter'.
        annotate_col: If `plot_type` is 'scatter', specifies a column name in `df` whose
            values will be used to annotate the scatter points. Requires values to be numeric.
        save_svg: If True and `out_file` is specified, saves an additional SVG version
            of the plot alongside the PNG.
    """

    num_axes = len(xs)
    if facet_vals is None:
        facet_vals = sorted(df[facet_by].unique())
    if sort_by is None and plot_type == "line":
        sort_by = y

    # TODO: For some reason the title is not centered at x=0.5. Fix
    xtitle_pos = 0.513

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig_width = 4 * num_axes
    fig_height_scale = 1.1 if len(facet_vals) == 1 else 1.0
    fig = plt.figure(figsize=(fig_width, fig_height_scale * 4 * len(facet_vals)), constrained_layout=True)
    subfigs = fig.subfigures(len(facet_vals))
    subfigs = np.atleast_1d(subfigs)  # type: ignore

    # Get all unique line values from the entire DataFrame
    all_line_vals = df[line_by].unique()
    if line_by_vals is not None:
        valid_line_by_vals = [val for val in line_by_vals if val in all_line_vals]
        if len(valid_line_by_vals) != len(line_by_vals):
            logger.warning(f"Some line_by_vals not found in data: {set(line_by_vals) - set(valid_line_by_vals)}")
        sorted_line_vals = valid_line_by_vals
        if not sorted_line_vals:
            logger.error(f"No valid line_by_vals found from {line_by_vals} in column {line_by}")
            plt.close(fig)
            return
    else:
        sorted_line_vals = sorted(all_line_vals, key=str)

    colors = sns.color_palette("tab10", n_colors=len(sorted_line_vals))
    # Create a map from line_val to its assigned color
    color_map = {val: color for val, color in zip(sorted_line_vals, colors)}

    # For figure legend
    legend_handles = {}

    for subfig, facet_val in zip(subfigs, facet_vals, strict=False):
        axs = subfig.subplots(1, num_axes)
        axs = np.atleast_1d(axs)  # Ensure axs is always array-like
        facet_df = df.loc[df[facet_by] == facet_val]

        # Iterate through each line value to plot it
        for line_val in sorted_line_vals:
            data = facet_df.loc[facet_df[line_by] == line_val]

            # Base style determined by line_by and plot_type
            base_color = color_map[line_val]  # Use the consistent color from the map
            base_linestyle = "-" if plot_type == "line" else "None"
            try:
                base_label = str(line_val)
            except Exception:
                base_label = "ErrorLabel"

            line_style = {
                "label": base_label,
                "marker": "o",
                "linewidth": 1.1,
                "color": base_color,
                "linestyle": base_linestyle,
            }
            # Apply overrides from 'styles' argument first
            line_style.update(
                {} if styles is None else styles.get(line_val, {})
            )
            current_label = line_style["label"]

            if not data.empty:
                plot_data = data.copy()
                if sort_by:
                    plot_data = plot_data.sort_values(sort_by)

                for i in range(num_axes):
                    x_col = xs[i]
                    if x_col not in plot_data.columns:
                        logger.warning(f"X-axis column '{x_col}' not found in data for line '{line_val}'. Skipping.")
                        continue
                    if y not in plot_data.columns:
                        logger.warning(f"Y-axis column '{y}' not found in data for line '{line_val}'. Skipping.")
                        continue

                    plot_data_ax = plot_data.dropna(subset=[x_col, y])
                    if annotate_col and annotate_col in plot_data_ax.columns:
                        plot_data_ax = plot_data_ax.dropna(subset=[annotate_col])

                    if plot_type == "scatter":
                        handle = axs[i].scatter(plot_data_ax[x_col], plot_data_ax[y], **line_style)
                        if annotate_col and not plot_data_ax.empty:
                            if annotate_col not in plot_data_ax.columns:
                                logger.warning(f"Annotation column '{annotate_col}' not found. Skipping annotations.")
                            else:
                                for _, point_data in plot_data_ax.iterrows():
                                    x_coord = point_data[x_col]
                                    y_coord = point_data[y]
                                    annotation_val = point_data[annotate_col]
                                    axs[i].text(
                                        x_coord,
                                        y_coord,
                                        f" {annotation_val:.2e}",
                                        fontsize=7,
                                        verticalalignment='bottom',
                                        horizontalalignment='left',
                                        clip_on=True
                                    )
                    elif plot_type == "line":
                        handle, = axs[i].plot(plot_data_ax[x_col], plot_data_ax[y], **line_style)
                    else:
                        raise ValueError(f"Unknown plot type: {plot_type}")

                    if current_label not in legend_handles:
                        legend_handles[current_label] = handle

            else:
                for i in range(num_axes):
                    handle, = axs[i].plot([], [], **line_style)
                    if current_label not in legend_handles:
                        legend_handles[current_label] = handle

        for i in range(num_axes):
            if xlims is not None and xlims[i] is not None:
                if facet_val in xlims[i]:
                    xmin, xmax = xlims[i][facet_val]
                    axs[i].set_xlim(xmin=xmin, xmax=xmax)
                else:
                    logger.warning(f"facet_val '{facet_val}' not found in xlims[{i}]. Using default limits.")
            if ylim is not None:
                if facet_val in ylim:
                    ymin, ymax = ylim[facet_val]
                    axs[i].set_ylim(ymin=ymin, ymax=ymax)
                else:
                    logger.warning(f"facet_val '{facet_val}' not found in ylim. Using default limits for axis {i}.")

            row_title = title.get(facet_val) if title is not None else f"{facet_by}={facet_val}"
            subfig.suptitle(row_title, fontweight="bold", x=xtitle_pos)

            axs[i].set_xlabel(xlabels[i] if xlabels is not None and i < len(xlabels) else xs[i])
            if i == 0:
                axs[i].set_ylabel(ylabel or y)
            else:
                axs[i].set_ylabel("")
                axs[i].set_yticklabels([])

            if xticks is not None and xticks[i] is not None:
                ticks, labels = xticks[i]
                axs[i].set_xticks(ticks, labels=labels)
            if yticks is not None and i == 0:
                axs[i].set_yticks(yticks[0], yticks[1])

        if axis_formatter is not None:
            axis_formatter(axs)

    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold", x=xtitle_pos)

    # Create figure legend
    if legend_handles:
        ordered_handles = [legend_handles[label] for label in legend_handles if label in legend_handles]
        ordered_labels = [label for label in legend_handles if label in legend_handles]

        fig.legend(
            ordered_handles,
            ordered_labels,
            title=legend_title or line_by,
            loc='upper right',
            bbox_to_anchor=(1, 1),
        )

    if out_file is not None:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(out_path, dpi=400, bbox_inches='tight')
            logger.info(f"Saved plot to {out_path}")
            if save_svg:
                svg_path = out_path.with_suffix(".svg")
                plt.savefig(svg_path, bbox_inches='tight')
                logger.info(f"Saved SVG plot to {svg_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {out_path}: {e}")

    plt.close(fig)
