#!/usr/bin/env python3
"""Reproduce the core numerical results in the Fourier Optics Lab report.

This script starts from raw CSV measurements, performs regression and
uncertainty propagation, and writes both machine-readable summaries
(CSV/JSON) and publication-ready plots.

The calculations mirror Appendix B of the report: screen-angle
regression, camera calibration, slit cutoff, and Abbe-limit estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "output"


@dataclass
class ScreenFitResult:
    slope_m: float  # tan(theta)
    slope_sigma: float
    theta_rad: float
    theta_sigma: float
    d_um: float
    d_sigma_um: float
    r2: float


@dataclass
class CameraResult:
    s_um_per_px: float
    s_sigma_um_per_px: float
    d_um: float
    d_sigma_um: float


@dataclass
class SlitResult:
    d_um: float
    d_sigma_um: float
    w_true_in: float
    delta_w_in: float
    delta_center_mm: float


@dataclass
class AbbeResult:
    NA: float
    dx_min_um: float


def ensure_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def through_origin_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = m x through the origin and return (m, sigma_m, r2)."""
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for regression.")

    m = float(np.sum(x * y) / np.sum(x * x))
    residuals = y - m * x
    sigma_m = math.sqrt(float(np.sum(residuals**2) / ((n - 1) * np.sum(x**2))))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum(y**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return m, sigma_m, r2


def screen_angle_method(lam_nm: float = 550.0) -> ScreenFitResult:
    df = pd.read_csv(DATA_RAW / "screen_angle.csv")
    L = df["L_cm"].to_numpy(dtype=float)
    y = df["y_cm"].to_numpy(dtype=float)

    m, sigma_m, r2 = through_origin_regression(L, y)

    theta = math.atan(m)
    sigma_theta = sigma_m / (1.0 + m**2)

    lam = lam_nm * 1e-9
    d = lam / math.sin(theta)
    sigma_d = abs(lam * math.cos(theta) / (math.sin(theta) ** 2)) * sigma_theta

    return ScreenFitResult(
        slope_m=m,
        slope_sigma=sigma_m,
        theta_rad=theta,
        theta_sigma=sigma_theta,
        d_um=d * 1e6,
        d_sigma_um=sigma_d * 1e6,
        r2=r2,
    )


def camera_method() -> CameraResult:
    df = pd.read_csv(DATA_RAW / "camera_calibration.csv").iloc[0]

    dx_um = float(df["cal_spacing_mm"]) * 1e3
    dp = float(df["cal_pixels_px"])
    sig_dp = float(df["sigma_cal_pixels_px"])

    N = float(df["count_pixels_px"])
    P = float(df["periods_count_P"])
    sig_P = float(df["sigma_periods_count_P"])

    s = dx_um / dp
    sig_s = s * sig_dp / dp

    d = s * N / P
    sig_d = d * math.sqrt((sig_s / s) ** 2 + (sig_P / P) ** 2)

    return CameraResult(
        s_um_per_px=s,
        s_sigma_um_per_px=sig_s,
        d_um=d,
        d_sigma_um=sig_d,
    )


def slit_cutoff_method(d_cam_um: float, lam_nm: float = 550.0) -> SlitResult:
    df = pd.read_csv(DATA_RAW / "slit_cutoff.csv").iloc[0]

    f = float(df["f_obj_mm"]) * 1e-3
    w_in = float(df["w_min_in"])
    sig_w_in = float(df["sigma_w_min_in"])

    inch = 0.0254
    w = w_in * inch
    sig_w = sig_w_in * inch

    lam = lam_nm * 1e-9
    d = 2.0 * f * lam / w
    sig_d = d * (sig_w / w)

    d_cam = d_cam_um * 1e-6
    w_true = 2.0 * f * lam / d_cam
    w_true_in = w_true / inch
    delta_w_in = w_true_in - w_in
    delta_center_mm = abs(delta_w_in) * inch * 1e3 / 2.0

    return SlitResult(
        d_um=d * 1e6,
        d_sigma_um=sig_d * 1e6,
        w_true_in=w_true_in,
        delta_w_in=delta_w_in,
        delta_center_mm=delta_center_mm,
    )


def abbe_limit(lam_nm: float = 550.0) -> AbbeResult:
    df = pd.read_csv(DATA_RAW / "abbe_params.csv").iloc[0]
    D = float(df["D_mm"]) * 1e-3
    f = float(df["f_mm"]) * 1e-3
    n = float(df["n"])

    lam = lam_nm * 1e-9
    NA = n * (D / (2.0 * f))
    dx_min = 0.61 * lam / NA
    return AbbeResult(NA=NA, dx_min_um=dx_min * 1e6)


def plot_screen_fit(screen: ScreenFitResult) -> None:
    df = pd.read_csv(DATA_RAW / "screen_angle.csv")
    L = df["L_cm"].to_numpy(dtype=float)
    y = df["y_cm"].to_numpy(dtype=float)

    L_line = np.linspace(0, L.max() * 1.05, 200)
    y_line = screen.slope_m * L_line

    plt.figure()
    plt.scatter(L, y, label="measured")
    plt.plot(L_line, y_line, label=f"fit: y = {screen.slope_m:.5f} L")
    plt.xlabel("Screen distance L (cm)")
    plt.ylabel("Order separation y (cm)")
    plt.title("Screen-angle method: y vs L")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "screen_fit.png", dpi=200)
    plt.close()

    plt.figure()
    residuals = y - screen.slope_m * L
    plt.axhline(0, linewidth=1)
    plt.scatter(L, residuals)
    plt.xlabel("Screen distance L (cm)")
    plt.ylabel("Residual y - mL (cm)")
    plt.title("Screen-angle method: residuals")
    plt.tight_layout()
    plt.savefig(OUT / "screen_residuals.png", dpi=200)
    plt.close()


def plot_uncertainty_budget(screen: ScreenFitResult, cam: CameraResult, slit: SlitResult) -> None:
    methods = ["screen", "camera", "slit (random)"]
    sigmas = [screen.d_sigma_um, cam.d_sigma_um, slit.d_sigma_um]

    plt.figure()
    plt.bar(methods, sigmas, color=["#4C78A8", "#54A24B", "#E45756"])
    plt.ylabel("Random uncertainty sigma_d (um)")
    plt.title("Experiment 1: random uncertainty comparison")
    plt.tight_layout()
    plt.savefig(OUT / "random_uncertainty_budget.png", dpi=200)
    plt.close()


def plot_grating_method_comparison(
    screen: ScreenFitResult, cam: CameraResult, slit: SlitResult
) -> None:
    methods = ["Screen angle", "Camera calibration", "Slit cutoff"]
    estimates = np.array([screen.d_um, cam.d_um, slit.d_um], dtype=float)
    sigmas = np.array([screen.d_sigma_um, cam.d_sigma_um, slit.d_sigma_um], dtype=float)
    ypos = np.arange(len(methods))

    plt.figure(figsize=(7.2, 3.8))
    plt.errorbar(
        estimates,
        ypos,
        xerr=sigmas,
        fmt="o",
        color="#1F77B4",
        ecolor="#1F77B4",
        capsize=4,
        markersize=7,
    )
    plt.axvline(
        cam.d_um,
        color="#54A24B",
        linestyle="--",
        linewidth=1.2,
        label="camera estimate",
    )
    plt.yticks(ypos, methods)
    plt.xlabel("Estimated grating period d (um)")
    plt.title("Experiment 1: independent grating-period estimates")
    plt.annotate(
        "slit width is systematic-dominated",
        xy=(slit.d_um + slit.d_sigma_um, ypos[2]),
        xytext=(11.15, ypos[2] + 0.35),
        arrowprops={"arrowstyle": "-", "color": "#666666"},
        fontsize=9,
        color="#444444",
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "grating_method_comparison.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()

    screen = screen_angle_method(lam_nm=550.0)
    cam = camera_method()
    slit = slit_cutoff_method(d_cam_um=cam.d_um, lam_nm=550.0)
    abbe = abbe_limit(lam_nm=550.0)

    summary = {
        "screen": asdict(screen),
        "camera": asdict(cam),
        "slit": asdict(slit),
        "abbe": asdict(abbe),
    }

    (DATA_PROCESSED / "results.json").write_text(json.dumps(summary, indent=2))

    rows = [
        {"method": "screen", "d_um": screen.d_um, "sigma_rand_um": screen.d_sigma_um},
        {"method": "camera", "d_um": cam.d_um, "sigma_rand_um": cam.d_sigma_um},
        {"method": "slit", "d_um": slit.d_um, "sigma_rand_um": slit.d_sigma_um},
    ]
    pd.DataFrame(rows).to_csv(DATA_PROCESSED / "grating_results.csv", index=False)

    plot_screen_fit(screen)
    plot_uncertainty_budget(screen, cam, slit)
    plot_grating_method_comparison(screen, cam, slit)

    print("=== Fourier Optics Lab: Reproduced results ===")
    print(f"Screen-angle: d = {screen.d_um:.2f} +/- {screen.d_sigma_um:.2f} um (random), R^2 = {screen.r2:.3f}")
    print(f"Camera:       d = {cam.d_um:.2f} +/- {cam.d_sigma_um:.2f} um (random)")
    print(f"Slit cutoff:  d = {slit.d_um:.2f} +/- {slit.d_sigma_um:.2f} um (random)")
    print(f"Abbe limit:   NA ~= {abbe.NA:.3f}, dx_min ~= {abbe.dx_min_um:.2f} um")
    print()
    print("Outputs written to:")
    print(f"  {DATA_PROCESSED}")
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
