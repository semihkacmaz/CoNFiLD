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

    # Add optional arguments
    parser.add_argument(
        "--N",
        type=str,
        default="fully_trained_44500",
        help="Checkpoint name",
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
        "--vals", type=int, default=3, help="Number of values to display per channel"
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
    parser.add_argument("--case", type=str, default="2", help="Cases 1, 2, 3, 4?")
    parser.add_argument(
        "--complete",
        type=bool,
        default=False,
        help="Flag to calculate statistics for the complete set",
    )

    args = parser.parse_args()

    N = args.N
    model_dir = args.model_dir
    data_dir = args.data_dir
    config_dir = args.config_dir
    case = args.case
    complete = args.complete

    # Form the specific field identifier and dataset name
    field = "Pub"
    dataset = "Pub"

    print(f"Model name: {N}")
    print(f"Using field: {field}")
    print(f"Using dataset: {dataset}")

    checkpoint_path = f"{model_dir}/{N}.pt"
    config_path = f"{config_dir}/case{case}.yml"
    data_path = f"{data_dir}/output.npy"

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
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        data_path=data_path,
        is_pub=True,
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
