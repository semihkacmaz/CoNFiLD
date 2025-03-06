import os
import glob
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from time import time


def calculate_integral_timescale(
    output_dir, output_dt, field, step_cutoff, u_component=0
):
    """
    Calculate the integral time scale from field data
    Disclaimer: All comments are added by an LLM for future reference.
    Comment accuracy not guaranteed.
    """
    start_time = time()
    field = field

    # Load and extract field data
    files = sorted(glob.glob(os.path.join(output_dir, "*.h5")))
    print(f"Found {len(files)} HDF5 files in {output_dir}")

    if not files:
        print(f"No .h5 files found in {output_dir}")
        return None

    with h5py.File(files[0], "r") as f:
        if f"tasks/{field}" not in f:
            print(f"No {field} data found in {files[0]}")
            return None

        field_data = f[f"tasks/{field}"][()]

        if field_data.shape[0] <= step_cutoff:
            raise ValueError(
                f"Not enough timesteps in data: {field_data.shape[0]} timesteps available, but {step_cutoff} timesteps required for cutoff"
            )

        timesteps_clipped = field_data.shape[0] - step_cutoff
        field_data = field_data[timesteps_clipped:]
        print(
            f"{field} field data shape: {field_data.shape} (Last {timesteps_clipped} timesteps clipped)"
        )

        # Extract time values if available -- just in case needed
        if "scales/sim_time" in f:
            time_values = f["scales/sim_time"][()]
        else:
            time_values = np.arange(field_data.shape[0]) * output_dt

    # Extract component of velocity for all points and all times
    # Assuming shape (time_steps, components, y, x)
    if len(field_data.shape) != 3 and len(field_data.shape) != 4:
        print(f"Unexpected field data shape: {field_data.shape}")
        return None

    if field == "velocity":
        u_field = field_data[:, u_component, :, :]  # Shape: [time, y, x]
    else:
        u_field = field_data

    n_times, ny, nx = u_field.shape
    print(f"Processing {field} field of shape {u_field.shape}")
    print(f"Time step between outputs is assumed: {output_dt:.2e}")

    if n_times < 4:
        print(f"Error: Not enough time points (need at least 4, got {n_times})")
        return None

    # Compute spatially averaged autocorrelation (vectorized)
    # Compute mean along time dimension for each spatial point
    u_mean = np.mean(u_field, axis=0)  # Shape: [ny, nx]

    # Normalize the velocity field
    u_fluctuations = u_field - u_mean[np.newaxis, :, :]  # Shape: [time, ny, nx]

    # Compute variance at each spatial point
    variance = np.var(u_field, axis=0)  # Shape: [ny, nx]
    valid_points = variance > 1e-10  # Mask for points with non-zero variance

    if not np.any(valid_points):
        print("Warning: Zero variance in all spatial points")
        return 0.0

    # Compute maximum lag
    max_lag = n_times // 2

    # Initialize autocorrelation array
    avg_autocorr = np.zeros(max_lag)

    # Vectorized autocorrelation calculation
    print("Computing autocorrelations...")
    for lag in range(max_lag):
        if lag == 0:
            avg_autocorr[lag] = 1.0
        else:
            # Element-wise multiply of lagged signals across all spatial points
            product = u_fluctuations[:-lag] * u_fluctuations[lag:]

            # Average over time for each spatial point
            spatial_autocorr = np.mean(product, axis=0) / (variance + 1e-10)

            # Average valid points for this lag
            avg_autocorr[lag] = np.mean(spatial_autocorr[valid_points])

    # Find appropriate cutoff
    if np.any(avg_autocorr < 0.05):
        cutoff_idx = np.argmax(avg_autocorr < 0.05)
    else:
        # Find local minima
        if max_lag > 3:
            minima_idx = (
                np.where(
                    (avg_autocorr[1:-1] < avg_autocorr[:-2])
                    & (avg_autocorr[1:-1] < avg_autocorr[2:])
                )[0]
                + 1
            )
            if len(minima_idx) > 0:
                cutoff_idx = minima_idx[0]  # Use first minimum
            else:
                cutoff_idx = max_lag // 2
        else:
            cutoff_idx = max_lag // 2

    # Ensure cutoff is at least 1
    cutoff_idx = max(1, cutoff_idx)

    # Integrate using trapezoidal rule
    time_lags = np.arange(cutoff_idx) * output_dt
    integral_timescale = np.trapz(avg_autocorr[:cutoff_idx], time_lags)

    end_time = time()
    print(f"Calculation completed in {end_time - start_time:.2f} seconds")

    integral_timesteps = int(integral_timescale // output_dt + 1.0)

    # Create plots
    plt.figure(figsize=(12, 8))

    # Plot autocorrelation
    plt.subplot(2, 1, 1)
    plt.plot(
        np.arange(max_lag) * output_dt,
        avg_autocorr,
        "b-",
        label="Spatial Avg Autocorrelation",
    )
    plt.axhline(y=0.05, color="r", linestyle="--", label="Threshold (0.05)")
    plt.axvline(
        x=cutoff_idx * output_dt,
        color="g",
        linestyle="-",
        label=f"Cutoff ({cutoff_idx * output_dt:.4f})",
    )
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.xlabel("Time lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"Spatially-Averaged {field} Autocorrelation - {Path(output_dir).name}")
    plt.grid(True)
    plt.legend()

    # Plot the integration area
    plt.subplot(2, 1, 2)
    plt.fill_between(time_lags, avg_autocorr[:cutoff_idx], alpha=0.3, color="b")
    plt.plot(time_lags, avg_autocorr[:cutoff_idx], "b-")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.xlabel("Time lag")
    plt.ylabel("Autocorrelation")
    plt.title(
        f"Integration Area - Integral Time Scale = {integral_timescale:.6f} s (~ {integral_timesteps} steps)"
    )
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"autocorrelation_detailed_{field}.png"))
    plt.close()

    return integral_timescale


def main():
    parser = argparse.ArgumentParser(
        description="Calculate integral time scales from MHD simulation data"
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="outputs_2",
        help="Base directory containing output-* folders",
    )
    parser.add_argument(
        "-dt", "--output_dt", type=float, default=1e-2, help="Time step between outputs"
    )
    parser.add_argument(
        "-s",
        "--specific",
        type=str,
        default=None,
        help="Process a specific output directory instead of all",
    )
    parser.add_argument(
        "-f",
        "--field",
        type=str,
        default="velocity",
        help="Field type? velocity or vector potential?",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=int,
        default=1200,
        help="Timestep cutoff (must be smaller than total number of timesteps in your simulation data)",
    )
    args = parser.parse_args()

    if args.specific:
        # Process a specific directory
        output_dir = args.specific
        if not os.path.isdir(output_dir):
            print(f"Directory not found: {output_dir}")
            return

        integral_ts = calculate_integral_timescale(
            output_dir, args.output_dt, args.field, args.cutoff
        )
        integral_timesteps = int(integral_ts // args.output_dt + 1.0)
        if integral_ts is not None:
            print(
                f"Integral time scale for the {args.field} field in {output_dir}: {integral_ts:.4f} s (~{integral_timesteps} steps)"
            )

            # Save result
            with open(
                os.path.join(output_dir, f"integral_timescale_{args.field}.txt"), "w"
            ) as f:
                f.write(f"Integral time scale (seconds): {integral_ts}\n")
                f.write(f"Integral time scale (steps): {integral_timesteps}\n")
    else:
        # Process all output directories
        output_dirs = sorted(glob.glob(os.path.join(args.data_dir, "output-*")))
        results = {}

        for output_dir in output_dirs:
            print(f"Processing {output_dir}...")
            integral_ts = calculate_integral_timescale(
                output_dir, args.output_dt, args.field, args.cutoff
            )

            if integral_ts is not None:
                results[output_dir] = integral_ts
                print(
                    f"Integral time scale for the {args.field} field: {integral_ts:.4f} (~{integral_timesteps} steps)"
                )

                # Save individual result
                with open(
                    os.path.join(output_dir, f"integral_timescale_{args.field}.txt"),
                    "w",
                ) as f:
                    f.write(f"Integral time scale: {integral_ts} s\n")
                    f.write(f"Integral time scale (steps): {integral_timesteps}\n")

        # Save summary
        if results:
            with open(
                os.path.join(args.data_dir, "integral_timescales_summary.txt"), "w"
            ) as f:
                for dir_name, ts in results.items():
                    f.write(f"{Path(dir_name).name}: {ts:.4f}\n")
                f.write(f"\nMean: {np.mean(list(results.values())):.4f}\n")
                f.write(f"Std: {np.std(list(results.values())):.4f}\n")

            print(f"\nProcessed {len(results)} directories")
            print(f"Mean integral time scale: {np.mean(list(results.values())):.4f}")
            print(
                f"Results summary saved to {os.path.join(args.data_dir, 'integral_timescales_summary.txt')}"
            )


if __name__ == "__main__":
    main()
