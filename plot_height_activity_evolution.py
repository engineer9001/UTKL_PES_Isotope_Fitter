#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split TPPT detector into 8 longitudinal sections (each 1/2 chip high), "
            "count same-height coincidence pairs, plot rate evolution, and fit isotope decays "
            "with fixed constant background."
        )
    )
    parser.add_argument("--dat-file", type=Path, required=True, help="Input coincidence .dat file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yuanwu/Library/CloudStorage/Box-Box/PET/PES_Isotope_Fitter/height_activity_output"),
        help=(
            "Base output directory (default: "
            "/Users/yuanwu/Library/CloudStorage/Box-Box/PET/PES_Isotope_Fitter/height_activity_output). "
            "A subfolder named after the input .dat file stem is created automatically."
        ),
    )
    parser.add_argument("--time-col", type=int, default=3, help="1-based TimeL column index (default: 3).")
    parser.add_argument(
        "--channel-cols",
        type=int,
        nargs=2,
        default=(5, 10),
        metavar=("IDL_COL", "IDR_COL"),
        help="1-based channel ID column indices (default: 5 10).",
    )
    parser.add_argument("--bin-width", type=float, default=1.0, help="Rate bin width in seconds (default: 1.0).")
    parser.add_argument(
        "--fit-isotopes",
        type=str,
        nargs="+",
        default=["C10", "C11", "N13", "O15"],
        help="Isotope names to fit (default: C10 C11 N13 O15).",
    )
    parser.add_argument(
        "--initial-background-rate",
        type=float,
        default=50.0,
        help="Initial guess of constant background rate in s^-1 for fitting (default: 200).",
    )
    parser.add_argument(
        "--background-lower-bound",
        type=float,
        default=0.0,
        help="Lower bound of fitted constant background rate (default: 0).",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1e12,
        help="Raw time values are divided by this factor to convert to seconds (default: 1e12).",
    )
    parser.add_argument(
        "--min-events-per-section",
        type=int,
        default=200,
        help="Minimum selected pair count in a section to run fit (default: 200).",
    )
    return parser.parse_args()


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_external_dicts(script_dir: Path):
    pes_dir = script_dir.parent / "PES_Isotope_Fitter"
    fit_params_path = pes_dir / "fit_params.py"
    channel_indices_path = pes_dir / "channel_indices.py"

    if not fit_params_path.exists():
        raise FileNotFoundError(f"Cannot find fit params file: {fit_params_path}")
    if not channel_indices_path.exists():
        raise FileNotFoundError(f"Cannot find channel indices file: {channel_indices_path}")

    fit_params_mod = _load_module(fit_params_path, "fit_params_dynamic")
    channel_indices_mod = _load_module(channel_indices_path, "channel_indices_dynamic")

    if not hasattr(fit_params_mod, "Isotopes_Lifetimes_Dict"):
        raise AttributeError("fit_params.py has no Isotopes_Lifetimes_Dict")
    if not hasattr(channel_indices_mod, "indices"):
        raise AttributeError("channel_indices.py has no indices mapping")

    return fit_params_mod.Isotopes_Lifetimes_Dict, channel_indices_mod.indices


def decode_abs_channel_id(abs_channel_id: np.ndarray):
    abs_channel_id = abs_channel_id.astype(np.int64, copy=False)
    port = abs_channel_id // 131072
    rem = abs_channel_id - 131072 * port
    slave = rem // 4096
    rem = rem - 4096 * slave
    chip = rem // 64
    channel = rem % 64
    return port, slave, chip, channel


def abs_to_geo_channel_id(abs_channel_id: np.ndarray, indices_map: dict[int, tuple[int, int]]) -> np.ndarray:
    port, slave, chip, channel = decode_abs_channel_id(abs_channel_id)
    pcb_chan = 64 * (chip % 2) + channel

    rc_lookup = np.array([indices_map[i] for i in range(128)], dtype=np.int64)
    r = rc_lookup[pcb_chan, 0]
    c = rc_lookup[pcb_chan, 1]
    abs_pcb_chan = 8 * r + c

    geo = 10**6 * port + 10**4 * slave + 10**2 * chip + (abs_pcb_chan % 64)
    return geo.astype(np.int64)


def crescent_row(geo_channel_id: np.ndarray) -> np.ndarray:
    geo_channel_id = geo_channel_id.astype(np.int64, copy=False)
    port = geo_channel_id // 10**6
    slave = (geo_channel_id - 10**6 * port) // 10**4
    chip = (geo_channel_id - slave * 10**4 - 10**6 * port) // 100
    channel = geo_channel_id % 100

    row = np.empty_like(chip)
    mask_low = chip < 8
    row[mask_low] = 8 * (chip[mask_low] % 2) + (channel[mask_low] // 8)
    row[~mask_low] = 16 + 8 * ((chip[~mask_low] - 8) % 2) + (channel[~mask_low] // 8)
    return row.astype(np.int64)


def make_fit_function(isotopes: list[str], taus: list[float]):
    param_names = [f"A_{iso}" for iso in isotopes] + ["ConstBkg"]

    def model(t, *params):
        t = np.asarray(t, dtype=float)
        amps = params[: len(isotopes)]
        const_bkg = params[-1]
        out = np.full_like(t, const_bkg, dtype=float)
        for amp, tau in zip(amps, taus):
            out += amp * np.exp(-t / tau)
        return out

    parameters = [inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in param_names
    ]
    model.__signature__ = inspect.Signature(parameters)
    return model, param_names


def fit_section_rates(
    bin_centers: np.ndarray,
    rates: np.ndarray,
    rate_err: np.ndarray,
    isotopes: list[str],
    taus: list[float],
    initial_background_rate: float,
    background_lower_bound: float,
):
    fit_func, param_names = make_fit_function(isotopes, taus)
    initial_amp = float(np.nanmax(rates) / max(len(isotopes), 1)) if rates.size else 1.0
    initial = {name: max(initial_amp, 1.0) for name in param_names}
    initial["ConstBkg"] = max(float(initial_background_rate), float(background_lower_bound))

    lsq = cost.LeastSquares(bin_centers, rates, rate_err, fit_func)
    fitter = Minuit(lsq, **initial)
    limits = [(0.0, None)] * len(isotopes) + [(background_lower_bound, None)]
    fitter.limits = limits
    fitter.migrad()
    fitter.hesse()

    ndf = len(rates) - fitter.nfit
    red_chi2 = float(fitter.fval / ndf) if ndf > 0 else np.nan

    params = {name: float(fitter.values[name]) for name in param_names}
    perrs = {name: float(fitter.errors[name]) for name in param_names}

    return {
        "ok": bool(fitter.valid),
        "fitter": fitter,
        "fit_func": fit_func,
        "params": params,
        "perrs": perrs,
        "chi2": float(fitter.fval),
        "ndf": int(ndf),
        "red_chi2": red_chi2,
    }


def main() -> None:
    args = parse_args()

    if args.bin_width <= 0:
        raise ValueError("--bin-width must be > 0")
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be > 0")
    if args.initial_background_rate < 0:
        raise ValueError("--initial-background-rate must be >= 0")
    if args.background_lower_bound < 0:
        raise ValueError("--background-lower-bound must be >= 0")

    dat_file = args.dat_file
    if not dat_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {dat_file}")

    out_dir = args.output_dir / dat_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    lifetime_dict, indices_map = load_external_dicts(Path(__file__).resolve().parent)
    missing = [iso for iso in args.fit_isotopes if iso not in lifetime_dict]
    if missing:
        raise ValueError(
            "Unknown isotope(s): "
            + ", ".join(missing)
            + ". Please use names defined in PES_Isotope_Fitter/fit_params.py"
        )

    usecols = [args.time_col - 1, args.channel_cols[0] - 1, args.channel_cols[1] - 1]
    if min(usecols) < 0:
        raise ValueError("Column indices are 1-based and must be >= 1")

    raw = np.loadtxt(dat_file, delimiter="\t", usecols=usecols)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    time_l = raw[:, 0].astype(np.float64) / args.time_scale
    id_l = raw[:, 1].astype(np.int64)
    id_r = raw[:, 2].astype(np.int64)

    finite_mask = np.isfinite(time_l)
    time_l = time_l[finite_mask]
    id_l = id_l[finite_mask]
    id_r = id_r[finite_mask]
    if time_l.size == 0:
        raise ValueError("No valid events found after time filtering")

    t = time_l - np.min(time_l)

    geo_l = abs_to_geo_channel_id(id_l, indices_map)
    geo_r = abs_to_geo_channel_id(id_r, indices_map)
    row_l = crescent_row(geo_l)
    row_r = crescent_row(geo_r)

    section_l = row_l // 4
    section_r = row_r // 4
    same_height_mask = section_l == section_r

    if not np.any(same_height_mask):
        raise ValueError(
            "No same-height events found (section_L == section_R). "
            "Check channel columns and input format."
        )

    t_sel = t[same_height_mask]
    sec_sel = section_l[same_height_mask]

    tmax = float(np.max(t_sel))
    edges = np.arange(0.0, tmax + args.bin_width, args.bin_width)
    if edges.size < 2:
        edges = np.array([0.0, args.bin_width], dtype=float)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    isotopes = list(args.fit_isotopes)
    taus = [float(lifetime_dict[iso]) for iso in isotopes]

    records_timeseries: list[dict[str, float | int]] = []
    records_fit: list[dict[str, float | int]] = []

    fig_data, axs_data = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    fig_fit, axs_fit = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    axs_data = axs_data.flatten()
    axs_fit = axs_fit.flatten()

    for section in range(8):
        ax_data = axs_data[section]
        ax_fit = axs_fit[section]

        section_mask = sec_sel == section
        section_times = t_sel[section_mask]
        pair_events_total = int(section_times.size)

        counts, _ = np.histogram(section_times, bins=edges)
        rates = counts.astype(float) / args.bin_width
        rate_err = np.sqrt(np.clip(counts.astype(float), 1.0, None)) / args.bin_width

        for k in range(len(bin_centers)):
            records_timeseries.append(
                {
                    "section_id": section,
                    "bin_left_s": float(edges[k]),
                    "bin_right_s": float(edges[k + 1]),
                    "bin_center_s": float(bin_centers[k]),
                    "count": int(counts[k]),
                    "rate_s-1": float(rates[k]),
                    "rate_err_s-1": float(rate_err[k]),
                }
            )

        ax_data.errorbar(bin_centers, rates, yerr=rate_err, fmt=".", markersize=3, linewidth=0.8)
        ax_data.set_title(f"Section {section}")
        ax_data.set_ylabel("Rate [s$^{-1}$]")
        ax_data.grid(alpha=0.2)

        ax_fit.errorbar(
            bin_centers,
            rates,
            yerr=rate_err,
            fmt=".",
            markersize=3,
            linewidth=0.8,
            label="Data",
            color="black",
            alpha=0.7,
            zorder=1,
        )
        ax_fit.set_title(f"Section {section}")
        ax_fit.set_ylabel("Rate [s$^{-1}$]")
        ax_fit.grid(alpha=0.2)

        fit_row: dict[str, float | int] = {
            "section_id": section,
            "pair_events_total": pair_events_total,
            "fit_used": 0,
            "fit_valid": 0,
            "chi2": np.nan,
            "ndf": np.nan,
            "reduced_chi2": np.nan,
            "background_rate_fit": np.nan,
            "background_rate_fit_uncertainty": np.nan,
        }
        for iso in isotopes:
            fit_row[f"A_{iso}"] = np.nan
            fit_row[f"Aerr_{iso}"] = np.nan

        if pair_events_total < args.min_events_per_section:
            ax_fit.text(
                0.97,
                0.97,
                f"Skip fit: N={pair_events_total} < {args.min_events_per_section}",
                transform=ax_fit.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            records_fit.append(fit_row)
            continue

        fit_res = fit_section_rates(
            bin_centers=bin_centers,
            rates=rates,
            rate_err=rate_err,
            isotopes=isotopes,
            taus=taus,
            initial_background_rate=args.initial_background_rate,
            background_lower_bound=args.background_lower_bound,
        )

        fit_row["fit_used"] = 1
        fit_row["fit_valid"] = int(fit_res["ok"])
        fit_row["chi2"] = float(fit_res["chi2"])
        fit_row["ndf"] = int(fit_res["ndf"])
        fit_row["reduced_chi2"] = float(fit_res["red_chi2"])
        fit_row["background_rate_fit"] = float(fit_res["params"]["ConstBkg"])
        fit_row["background_rate_fit_uncertainty"] = float(fit_res["perrs"]["ConstBkg"])

        dense_t = np.linspace(bin_centers.min(), bin_centers.max(), max(200, 8 * len(bin_centers)))
        total = np.full_like(dense_t, fit_row["background_rate_fit"], dtype=float)
        ax_fit.axhline(
            fit_row["background_rate_fit"],
            linestyle=":",
            color="gray",
            linewidth=1.5,
            label="ConstBkg (fit)",
            zorder=2,
        )

        for iso, tau in zip(isotopes, taus):
            amp_name = f"A_{iso}"
            amp_val = float(fit_res["params"][amp_name])
            amp_err = float(fit_res["perrs"][amp_name])
            fit_row[amp_name] = amp_val
            fit_row[f"Aerr_{iso}"] = amp_err

            comp = amp_val * np.exp(-dense_t / tau)
            total += comp
            ax_fit.plot(dense_t, comp, linewidth=1.6, label=iso, zorder=3)

        ax_fit.plot(dense_t, total, color="red", linewidth=2.2, label="Sum", zorder=4)
        ax_fit.text(
            0.97,
            0.97,
            f"valid={fit_res['ok']}\n$\\chi^2$/ndf={fit_res['red_chi2']:.2f}",
            transform=ax_fit.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        records_fit.append(fit_row)

    for idx, ax in enumerate(axs_data):
        if idx >= 4:
            ax.set_xlabel("Time [s]")
    for idx, ax in enumerate(axs_fit):
        if idx >= 4:
            ax.set_xlabel("Time [s]")

    handles: list = []
    labels: list[str] = []
    for ax in axs_fit:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                labels.append(ll)
                handles.append(hh)
    if handles:
        fig_fit.legend(handles, labels, loc="upper right", fontsize=9)

    fig_data.suptitle("Events Rate Evolution by Height Section (same-height pairs)", fontsize=14)
    fig_fit.suptitle("Events Rate + Isotope Fits by Height Section", fontsize=14)
    fig_data.tight_layout(rect=(0, 0, 1, 0.98))
    fig_fit.tight_layout(rect=(0, 0, 1, 0.98))

    out_rate_png = out_dir / "height_section_rate_evolution.png"
    out_fit_png = out_dir / "height_section_rate_fit.png"
    out_depth_png = out_dir / "height_section_isotope_depth_profile.png"
    out_fit_tsv = out_dir / "height_section_fit_results.tsv"
    out_ts_tsv = out_dir / "height_section_rate_timeseries.tsv"

    fig_data.savefig(out_rate_png, dpi=200)
    fig_fit.savefig(out_fit_png, dpi=200)
    axial_length_mm = 105.0
    section_height_mm = axial_length_mm / 8.0
    depth_mm = np.array([(i + 0.5) * section_height_mm for i in range(8)], dtype=float)
    fit_df = pd.DataFrame(records_fit).sort_values("section_id")

    fig_depth, ax_depth = plt.subplots(figsize=(8, 5))
    for iso in isotopes:
        y = fit_df[f"A_{iso}"].to_numpy(dtype=float)
        yerr = fit_df[f"Aerr_{iso}"].to_numpy(dtype=float)
        ax_depth.errorbar(
            depth_mm,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            markersize=5,
            capsize=3,
            label=iso,
        )
    ax_depth.set_xlabel("Depth from top [mm]")
    ax_depth.set_ylabel("Fitted isotope rate amplitude [s$^{-1}$]")
    ax_depth.set_title("Isotope Rate vs Axial Depth (105 mm)")
    ax_depth.set_xlim(0, axial_length_mm)
    ax_depth.grid(alpha=0.25)
    ax_depth.legend()
    fig_depth.tight_layout()
    fig_depth.savefig(out_depth_png, dpi=220)

    plt.close(fig_data)
    plt.close(fig_fit)
    plt.close(fig_depth)

    pd.DataFrame(records_fit).to_csv(out_fit_tsv, sep="\t", index=False)
    pd.DataFrame(records_timeseries).to_csv(out_ts_tsv, sep="\t", index=False)

    print(f"Saved: {out_rate_png}")
    print(f"Saved: {out_fit_png}")
    print(f"Saved: {out_depth_png}")
    print(f"Saved: {out_fit_tsv}")
    print(f"Saved: {out_ts_tsv}")


if __name__ == "__main__":
    main()
