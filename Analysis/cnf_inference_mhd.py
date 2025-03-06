from cnf.inference_function import CNF_inference
import numpy as np
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="CNF inference script for data evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add required arguments
    parser.add_argument(
        "sim_end_time", type=str, help="Simulation end time (e.g. '1p5')"
    )
    parser.add_argument("field_id", type=str, help="Field identifier (e.g. 'A')")

    # Add optional arguments
    parser.add_argument(
        "--N",
        type=str,
        default="50000",
        help="Specific checkpoint to load (e.g.'checkpoint_N.pt' ",
    )
    parser.add_argument(
        "--latent_indices",
        type=int,
        nargs="+",
        default=[0, 599, 1199],
        help="List of latent indices for prediction",
    )
    parser.add_argument(
        "--timestep", type=int, default=0, help="Timestep index for comparison"
    )
    parser.add_argument("--row", type=int, default=0, help="Row index for comparison")
    parser.add_argument(
        "--vals", type=int, default=5, help="Number of values to display per channel"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/path/to/model",
        help="Checkpoint directory to load from",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/data",
        help="Data directory to load from",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="/path/to/config",
        help="Config directory to load from",
    )
    parser.add_argument(
        "--complete",
        type=bool,
        default=False,
        help="Flag to calculate statistics for the complete set",
    )

    args = parser.parse_args()

    sim_end_time = args.sim_end_time
    field_id = args.field_id
    N = args.N
    model_dir = args.model_dir
    data_dir = args.data_dir
    config_dir = args.config_dir
    complete = args.complete

    if not sim_end_time:
        print("ERROR: Simulation end time cannot be empty.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if not field_id:
        print("ERROR: Field identifier cannot be empty.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Form the specific field identifier and dataset name
    field = f"{sim_end_time}_{field_id}"
    dataset = f"uxuy_T{field}"

    print(f"Checkpoint: {N}")
    print(f"Using field: {field}")
    print(f"Using dataset: {dataset}")

    checkpoint_path = f"{model_dir}/{dataset}/checkpoint_{N}.pt"
    config_path = f"{config_dir}/case2_{field}.yml"
    data_path = f"{data_dir}/{dataset}/{dataset}.npy"

    for path, name in [
        (checkpoint_path, "Checkpoint"),
        (config_path, "Config"),
        (data_path, "Data"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    print("Initializing CNF inference...")
    infer = CNF_inference(
        checkpoint_path=checkpoint_path, config_path=config_path, data_path=data_path
    )

    coords = infer.create_coordinates_grid()

    latent_indices = args.latent_indices
    Nt = args.timestep
    row = args.row
    vals = args.vals

    if Nt not in latent_indices:
        raise ValueError(f"Timestep {Nt} not found in latent indices {latent_indices}")

    Nt_idx = latent_indices.index(Nt)

    print(f"Generating predictions for latent indices {latent_indices}...")
    predictions = infer.predict(coords, latent_indices=latent_indices)
    print(predictions.shape)

    # Load data for comparison
    data = np.load(data_path)
    num_channels = predictions.shape[-1]

    print(f"\n{'=' * 50}")
    print(f"COMPARISON AT TIMESTEP {Nt}, ROW {row}, FIRST {vals} COLUMNS")
    print(f"{'=' * 50}")

    total_mse = 0
    total_rmae = 0

    for c in range(num_channels):
        coord_info = f"Channel {c}"

        print(f"\n{'-' * 50}")
        print(f"{coord_info}")
        print(f"{'-' * 50}")

        print(f"DATA:       {data[Nt, row, :vals, c]}")
        print(f"PREDICTION: {predictions[Nt_idx, row, :vals, c].numpy()}")

        # Calculate and show error
        error = np.abs(
            data[Nt, row, :vals, c] - predictions[Nt_idx, row, :vals, c].numpy()
        )
        print(f"ABS ERROR:  {error}")
        print(f"MEAN ERROR: {np.mean(error):.3e}")

        # Calculate MSE and RMAE for this channel
        channel_mse = np.mean(
            (data[Nt, row, :vals, c] - predictions[Nt_idx, row, :vals, c].numpy()) ** 2
        )
        channel_rmae = np.mean(
            np.abs(data[Nt, row, :vals, c] - predictions[Nt_idx, row, :vals, c].numpy())
            / (np.abs(data[Nt, row, :vals, c]) + 1e-8)
        )

        print(f"CHANNEL MSE:  {channel_mse:.3e}")
        print(f"CHANNEL RMAE: {channel_rmae:.3e}")

        total_mse += channel_mse
        total_rmae += channel_rmae

    # Calculate and display average metrics across all channels
    avg_mse = total_mse / num_channels
    avg_rmae = total_rmae / num_channels

    print(f"\n{'=' * 50}")
    print(f"OVERALL METRICS ACROSS ALL CHANNELS")
    print(f"{'=' * 50}")
    print(f"AVERAGE MSE:  {avg_mse:.3e}")
    print(f"AVERAGE RMAE: {avg_rmae:.3e}")

    if complete:
        print("\nNow, calculating the statistics for the entire set.")
        print("This may take a while...")
        all_predictions = infer.get_all_predictions(coords)
        for c in range(num_channels):
            coord_info = f"Channel {c}"

            print(f"\n{'-' * 50}")
            print(f"{coord_info}")
            print(f"{'-' * 50}")

            # Calculate and show error
            error = np.abs(data[:1200, :, :, c] - all_predictions[:, :, :, c].numpy())
            print(f"MEAN ERROR: {np.mean(error):.3e}")

            # Calculate MSE and RMAE for this channel
            channel_mse = np.mean(
                (data[:1200, :, :, c] - all_predictions[:, :, :, c].numpy()) ** 2
            )
            channel_rmae = np.mean(
                np.abs(data[:1200, :, :, c] - all_predictions[:, :, :, c].numpy())
                / (np.abs(data[:1200, :, :, c]) + 1e-8)
            )

            print(f"CHANNEL MSE:  {channel_mse:.3e}")
            print(f"CHANNEL RMAE: {channel_rmae:.3e}")

            total_mse += channel_mse
            total_rmae += channel_rmae

        # Calculate and display average metrics across all channels
        avg_mse = total_mse / num_channels
        avg_rmae = total_rmae / num_channels
        print(f"\n{'=' * 50}")
        print(f"OVERALL METRICS ACROSS ALL CHANNELS")
        print(f"{'=' * 50}")
        print(f"AVERAGE MSE:  {avg_mse:.3e}")
        print(f"AVERAGE RMAE: {avg_rmae:.3e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
