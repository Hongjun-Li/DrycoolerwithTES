from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


MAT_FILE = Path(__file__).with_name("Chiller_Less_IBM_11_new.mat")
WINDOW_START_H = 5176.0
WINDOW_END_H = 5198.0

# ASHRAE Guideline 14 commonly used whole-building hourly calibration criteria.
# Guideline 14 is for energy model calibration; temperature is reported here with
# the same metrics, but the pass/fail criterion is directly applicable to energy.
ASHRAE_HOURLY_NMBE_LIMIT = 10.0
ASHRAE_HOURLY_CVRMSE_LIMIT = 30.0


def decode_char_matrix(value):
    """Decode a Dymola MATLAB char matrix into one string per variable."""
    arr = np.asarray(value)
    if arr.ndim == 1:
        return ["".join(arr).strip()]
    return ["".join(arr[:, i]).strip() for i in range(arr.shape[1])]


def load_dymola_result(mat_file):
    mat = loadmat(mat_file, squeeze_me=False, chars_as_strings=False)
    names = decode_char_matrix(mat["name"])
    data_info = np.asarray(mat["dataInfo"], dtype=int)
    data_2 = np.asarray(mat["data_2"], dtype=float)

    result = {}
    for i, name in enumerate(names):
        table_id = data_info[0, i]
        row_id = data_info[1, i]
        sign = -1.0 if row_id < 0 else 1.0
        row_idx = abs(row_id) - 1

        if table_id == 0:
            if name == "Time":
                result[name] = data_2[row_idx, :]
            continue
        if table_id != 2:
            raise ValueError(f"Unsupported Dymola data table {table_id} for {name}")

        result[name] = sign * data_2[row_idx, :]

    return result


def screenshot_reference_data():
    rows = [
        ("8/4/2011", "4:00:30 PM", 0.440, 36.3),
        ("8/4/2011", "4:59:56 PM", 0.442, 36.4),
        ("8/4/2011", "6:00:22 PM", 0.441, 36.4),
        ("8/4/2011", "7:00:54 PM", 0.440, 36.2),
        ("8/4/2011", "8:01:25 PM", 0.433, 35.8),
        ("8/4/2011", "9:01:54 PM", 0.433, 35.2),
        ("8/4/2011", "10:02:17 PM", 0.438, 34.2),
        ("8/4/2011", "11:02:41 PM", 0.437, 33.3),
        ("8/5/2011", "12:03:04 AM", 0.439, 32.6),
        ("8/5/2011", "1:03:27 AM", 0.440, 32.0),
        ("8/5/2011", "2:03:49 AM", 0.443, 31.2),
        ("8/5/2011", "3:04:12 AM", 0.441, 30.8),
        ("8/5/2011", "4:04:34 AM", 0.442, 30.4),
        ("8/5/2011", "5:04:58 AM", 0.443, 30.3),
        ("8/5/2011", "6:05:24 AM", 0.448, 30.4),
        ("8/5/2011", "7:05:46 AM", 0.440, 30.4),
        ("8/5/2011", "8:06:10 AM", 0.437, 31.1),
        ("8/5/2011", "9:06:41 AM", 0.432, 33.3),
        ("8/5/2011", "10:07:10 AM", 0.432, 35.0),
        ("8/5/2011", "11:07:40 AM", 0.437, 35.9),
        ("8/5/2011", "12:08:12 PM", 0.449, 36.6),
        ("8/5/2011", "1:08:44 PM", 0.465, 37.3),
        ("8/5/2011", "1:55:08 PM", 0.477, 37.6),
    ]
    ref = pd.DataFrame(
        rows,
        columns=[
            "date",
            "time_stamp",
            "measured_cooling_power_kw",
            "measured_rack_inlet_water_temp_c",
        ],
    )
    timestamps = pd.to_datetime(
        ref["date"] + " " + ref["time_stamp"],
        format="%m/%d/%Y %I:%M:%S %p",
    )
    elapsed_hours = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    ref.insert(2, "elapsed_hours", elapsed_hours)
    ref.insert(3, "aligned_time_hours", WINDOW_START_H + elapsed_hours)
    return ref


def to_celsius(values):
    values = np.asarray(values, dtype=float)
    return values - 273.15 if np.nanmean(values) > 200.0 else values


def validation_metrics(measured, predicted):
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mean_measured = np.mean(measured)
    residual = measured - predicted
    nmbe = np.sum(residual) / (len(measured) * mean_measured) * 100.0
    cvrmse = np.sqrt(np.mean(residual**2)) / mean_measured * 100.0
    return nmbe, cvrmse


def ashrae_hourly_pass(nmbe, cvrmse):
    return (
        abs(nmbe) <= ASHRAE_HOURLY_NMBE_LIMIT
        and cvrmse <= ASHRAE_HOURLY_CVRMSE_LIMIT
    )


def build_comparison():
    result = load_dymola_result(MAT_FILE)
    time_h = result["Time"] / 3600.0
    mat_power_kw = result["yPHVAC"] / 1000.0
    mat_rack_water_temp_c = to_celsius(result["TRacIn.T"])

    ref = screenshot_reference_data()
    ref["predicted_cooling_power_kw"] = np.interp(
        ref["aligned_time_hours"], time_h, mat_power_kw
    )
    ref["predicted_rack_inlet_water_temp_c"] = np.interp(
        ref["aligned_time_hours"], time_h, mat_rack_water_temp_c
    )
    ref["cooling_power_residual_kw"] = (
        ref["measured_cooling_power_kw"] - ref["predicted_cooling_power_kw"]
    )
    ref["rack_water_temp_residual_c"] = (
        ref["measured_rack_inlet_water_temp_c"]
        - ref["predicted_rack_inlet_water_temp_c"]
    )
    ref["rack_water_temp_residual_percent"] = (
        ref["rack_water_temp_residual_c"]
        / ref["measured_rack_inlet_water_temp_c"]
        * 100.0
    )

    return ref, time_h, mat_power_kw, mat_rack_water_temp_c


def make_metric_table(ref):
    rows = []
    metric_specs = [
        (
            "Cooling power",
            "measured_cooling_power_kw",
            "predicted_cooling_power_kw",
            True,
        ),
        (
            "Inlet rack temperature",
            "measured_rack_inlet_water_temp_c",
            "predicted_rack_inlet_water_temp_c",
            False,
        ),
    ]

    for variable, measured_col, predicted_col, ashrae_directly_applicable in metric_specs:
        nmbe, cvrmse = validation_metrics(ref[measured_col], ref[predicted_col])
        if ashrae_directly_applicable:
            status = "PASS" if ashrae_hourly_pass(nmbe, cvrmse) else "FAIL"
        else:
            status = "Reported only"
        rows.append(
            {
                "Variable": variable,
                "NMBE [%]": nmbe,
                "CVRMSE [%]": cvrmse,
                "ASHRAE Guideline 14 hourly check": status,
            }
        )

    return pd.DataFrame(rows)


def plot_validation_figure(ref, time_h, mat_rack_water_temp_c):
    mask = (time_h >= WINDOW_START_H) & (time_h <= WINDOW_END_H)
    fig = plt.figure(figsize=(12, 4.2))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.35, 1.0),
        wspace=0.28,
    )
    right = outer[0, 1].subgridspec(
        1,
        2,
        width_ratios=(1.0, 0.22),
        wspace=0.08,
    )

    ax_temp = fig.add_subplot(outer[0, 0])
    ax_temp.plot(
        time_h[mask],
        mat_rack_water_temp_c[mask],
        label="Simulation",
        color="#2ca02c",
        lw=1.8,
    )
    ax_temp.scatter(
        ref["aligned_time_hours"],
        ref["measured_rack_inlet_water_temp_c"],
        label="Reference",
        color="#7f7f7f",
        s=30,
        zorder=3,
    )
    ax_temp.set_xlabel("Aligned time [h]")
    ax_temp.set_ylabel("Rack inlet water temp [degC]")
    ax_temp.grid(True, linestyle=":", alpha=0.6)
    ax_temp.legend(loc="best")

    simulated = ref["predicted_rack_inlet_water_temp_c"].to_numpy()
    residual_percent = ref["rack_water_temp_residual_percent"].to_numpy()
    within_limit = np.abs(residual_percent) <= 5.0

    ax = fig.add_subplot(right[0, 0])
    ax_hist = fig.add_subplot(right[0, 1], sharey=ax)
    ax.scatter(
        simulated[within_limit],
        residual_percent[within_limit],
        s=38,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Residuals within ±5%",
    )
    ax.scatter(
        simulated[~within_limit],
        residual_percent[~within_limit],
        s=38,
        color="#d62728",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Residuals outside ±5%",
    )
    ax.axhline(0.0, color="#1f77b4", lw=1.2, ls="--", label="Ideal")
    ax.axhline(5.0, color="#ff9900", lw=1.0, ls="--", label="±5%")
    ax.axhline(-5.0, color="#ff9900", lw=1.0, ls="--")
    ax.set_xlabel("Simulation Temperature [degC]")
    ax.set_ylabel("Residual [%]")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", fontsize="small")

    ax_hist.hist(
        residual_percent,
        bins=10,
        orientation="horizontal",
        color="#9e9e9e",
        alpha=0.75,
        edgecolor="white",
        rwidth=0.72,
    )
    ax_hist.axhline(0.0, color="#1f77b4", lw=1.2, ls="--")
    ax_hist.axhline(5.0, color="#ff9900", lw=1.0, ls="--")
    ax_hist.axhline(-5.0, color="#ff9900", lw=1.0, ls="--")
    ax_hist.set_xlabel("Count")
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(True, axis="x", linestyle=":", alpha=0.3)

    out_png = Path(__file__).with_name("validationIBM_chillerless_rack_temp_validation.png")
    out_svg = Path(__file__).with_name("validationIBM_chillerless_rack_temp_validation.svg")
    out_pdf = Path(__file__).with_name("validationIBM_chillerless_rack_temp_validation.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    return out_png, out_svg, out_pdf


def plot_residual_figure(ref):
    simulated = ref["predicted_rack_inlet_water_temp_c"].to_numpy()
    residual_percent = ref["rack_water_temp_residual_percent"].to_numpy()
    within_limit = np.abs(residual_percent) <= 5.0

    fig = plt.figure(figsize=(7.0, 4.6))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.20),
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[0, 1], sharey=ax)

    ax.scatter(
        simulated[within_limit],
        residual_percent[within_limit],
        s=38,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Residuals within ±5%",
    )
    ax.scatter(
        simulated[~within_limit],
        residual_percent[~within_limit],
        s=38,
        color="#d62728",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Residuals outside ±5%",
    )
    ax.axhline(0.0, color="#1f77b4", lw=1.2, ls="--", label="Ideal")
    ax.axhline(5.0, color="#ff9900", lw=1.0, ls="--", label="±5%")
    ax.axhline(-5.0, color="#ff9900", lw=1.0, ls="--")
    ax.set_xlabel("Simulation Temperature [degC]", fontsize=12)
    ax.set_ylabel("Residual [%]", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        loc="upper right",
        ncol=1,
        fontsize=10,
        frameon=True,
        handletextpad=0.6,
    )

    ax_hist.hist(
        residual_percent,
        bins=10,
        orientation="horizontal",
        color="#9e9e9e",
        alpha=0.75,
        edgecolor="white",
        rwidth=0.72,
    )
    ax_hist.axhline(0.0, color="#1f77b4", lw=1.2, ls="--")
    ax_hist.axhline(5.0, color="#ff9900", lw=1.0, ls="--")
    ax_hist.axhline(-5.0, color="#ff9900", lw=1.0, ls="--")
    ax_hist.set_xlabel("Count", fontsize=12)
    ax_hist.tick_params(axis="x", labelsize=10)
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(True, axis="x", linestyle=":", alpha=0.3)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.96)
    out_png = Path(__file__).with_name("validationIBM_chillerless_rack_temp_residuals.png")
    out_svg = Path(__file__).with_name("validationIBM_chillerless_rack_temp_residuals.svg")
    out_pdf = Path(__file__).with_name("validationIBM_chillerless_rack_temp_residuals.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    return out_png, out_svg, out_pdf


def main():
    ref, time_h, mat_power_kw, mat_rack_water_temp_c = build_comparison()
    metric_table = make_metric_table(ref)

    out_csv = Path(__file__).with_name("validationIBM_chillerless_comparison.csv")
    out_metrics = Path(__file__).with_name("validationIBM_chillerless_metrics.csv")
    ref.to_csv(out_csv, index=False)
    metric_table.to_csv(out_metrics, index=False)
    out_png, out_svg, out_pdf = plot_validation_figure(ref, time_h, mat_rack_water_temp_c)
    residual_png, residual_svg, residual_pdf = plot_residual_figure(ref)

    print(f"MAT file: {MAT_FILE}")
    print(f"Comparison window: {WINDOW_START_H:g}-{WINDOW_END_H:g} h")
    print(f"Reference points: {len(ref)}")
    print()
    print(metric_table.to_string(index=False, formatters={
        "NMBE [%]": "{:.2f}".format,
        "CVRMSE [%]": "{:.2f}".format,
    }))
    print()
    print(
        "ASHRAE Guideline 14 hourly energy calibration criterion used: "
        f"|NMBE| <= {ASHRAE_HOURLY_NMBE_LIMIT:.0f}% and "
        f"CVRMSE <= {ASHRAE_HOURLY_CVRMSE_LIMIT:.0f}%."
    )
    print("Temperature is not an energy-use variable, so it is reported with the same metrics only.")
    print()
    print(f"Saved comparison table: {out_csv}")
    print(f"Saved metric table: {out_metrics}")
    print(f"Saved validation figure: {out_png}")
    print(f"Saved SVG validation figure: {out_svg}")
    print(f"Saved PDF validation figure: {out_pdf}")
    print(f"Saved residual figure: {residual_png}")
    print(f"Saved SVG residual figure: {residual_svg}")
    print(f"Saved PDF residual figure: {residual_pdf}")


if __name__ == "__main__":
    main()
