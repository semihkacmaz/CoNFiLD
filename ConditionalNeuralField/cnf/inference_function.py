import numpy as np
import torch
import sys
import os
import yaml

# Not the best approach overall, but it's the best one for current purposes
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# Import directly from the right location
from scripts.train import LatentContainer


def ReconstructFrame(data, mask, shape, fill_value=np.nan):
    temp_data = np.empty((*shape, data.shape[-1]))
    temp_data[:] = fill_value
    temp_data[mask] = data
    return temp_data


def pass_through_model_batch(
    coords, latents, model, x_normalizer, y_normalizer, batch_size, device
):
    t_size, latent_size = latents.shape
    m_size, coords_size = coords.shape
    if t_size % batch_size == 0:
        num_batches = t_size // batch_size
    else:
        num_batches = t_size // batch_size + 1
    output_all = []
    for i in range(num_batches):
        sid = int(i * batch_size)
        if i < num_batches - 1:
            eid = int((i + 1) * batch_size)
        else:
            eid = int(t_size)
        batch_latent = latents[sid:eid].reshape(-1, 1, latent_size)
        batch_coords = coords.reshape(1, m_size, coords_size).to(
            device
        )  # <1, meshsize, cin>
        batch_output = y_normalizer.denormalize(
            model(x_normalizer.normalize(batch_coords), batch_latent)
        )
        # <batch, meshsize, cout>
        output_all.append(batch_output)
    output_all = torch.cat(output_all, dim=0)
    return output_all


def decoder(coords, latents, model, x_normalizer, y_normalizer, batch_size, device):
    t_size, latent_size = latents.shape
    m_size, coords_size = coords.shape
    if t_size % batch_size == 0:
        num_batches = t_size // batch_size
    else:
        num_batches = t_size // batch_size + 1
    output_all = []
    with torch.no_grad():
        for i in range(num_batches):
            sid = int(i * batch_size)
            if i < num_batches - 1:
                eid = int((i + 1) * batch_size)
            else:
                eid = int(t_size)
            batch_latent = latents[sid:eid].reshape(-1, 1, latent_size)
            batch_coords = coords.reshape(1, m_size, coords_size).to(
                device
            )  # <1, meshsize, cin>
            batch_output = y_normalizer.denormalize(
                model(x_normalizer.normalize(batch_coords), batch_latent)
            )
            # <batch, meshsize, cout>
            output_all.append(batch_output.cpu())
        output_all = torch.cat(output_all, dim=0)
    return output_all


class CNF_inference:
    """Class for making inferences with trained Conditional Neural Field models.

    This class provides functionality to load trained CNF models and generate
    predictions using their latent codes.

    Example usage:
    -------------
    ```python
    from cnf.inference_function import CNF_inference

    checkpoint_path = (
        "/path/to/checkpoint"
    )
    config_path = (
        "/path/to/config"
    )
    data_path = "/path/to/data"

    infer = CNF_inference(
        checkpoint_path=checkpoint_path, config_path=config_path, data_path=data_path
    )

    # Create coordinates grid using data shape
    coords = infer.create_coordinates_grid()

    # Generate predictions
    predictions = infer.predict(coords, latent_indices=[0, 5, 10])

    # Or for all snapshots
    all_predictions = infer.get_all_predictions(coords)
    ```
    """

    def __init__(
        self, checkpoint_path, config_path, data_path, device="cuda", is_pub=False
    ):
        """
        Initialize inference with a trained CNF model.

        Args:
            checkpoint_path: Path to model checkpoint file
            config_path: Path to configuration file (YAML)
            data_path: Path to data file
            device: Device to run inference on ('cuda' or 'cpu')
        """

        for path, name in [
            (checkpoint_path, "checkpoint"),
            (config_path, "config"),
            (data_path, "data"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name.capitalize()} file not found at {path}")

        self.is_pub = is_pub  # Are you working with the model provided by the authors?
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.checkpoint_path = checkpoint_path

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data = np.load(data_path)
        print(f"Data loaded from {data_path}, shape: {self.data.shape}")

        norm_path = f"{os.path.dirname(checkpoint_path)}/normalizer_params.pt"
        if os.path.exists(norm_path):
            self.norm_params = torch.load(norm_path)
            from cnf.utils.normalize import Normalizer_ts

            self.x_normalizer = Normalizer_ts()
            self.y_normalizer = Normalizer_ts()
            self.x_normalizer.params = self.norm_params["x_normalizer_params"]
            self.y_normalizer.params = self.norm_params["y_normalizer_params"]
        else:
            raise FileNotFoundError(
                f"Normalizer parameters not found at {norm_path}. Cannot proceed with inference."
            )

        self._load_model()

    def _load_model(self):
        """Load the model and latents from checkpoint and config."""
        # Import dynamically based on config
        from cnf import nf_networks

        # Get model parameters from config
        nf_config = self.config.get("NF", {})
        model_type = nf_config.get("name", "SIRENAutodecoder_film")
        model_params = {k: v for k, v in nf_config.items() if k != "name"}

        # Handle special case for kwargs
        if "kwargs" in nf_config:
            model_params = nf_config["kwargs"]

        if not hasattr(nf_networks, model_type):
            raise ValueError(f"Model type {model_type} not found in nf_networks module")

        ModelClass = getattr(nf_networks, model_type)

        # Get latent dimensions from checkpoint -- trained model provided by the authors is structurally different
        if self.is_pub:
            latent_params = self.checkpoint["hidden_states"]
        else:
            latent_params = self.checkpoint["hidden_states"].get("latents")
        if latent_params is not None:
            N_samples, N_features = latent_params.shape
            model_params.setdefault("in_latent_features", N_features)
        else:
            raise ValueError("Could not find latent codes in checkpoint")

        self.model = ModelClass(**model_params).to(self.device)

        # Get dimensions for latent container
        dims = self.config.get("dims", 2)
        lumped = self.config.get("lumped_latent", False)

        self.latents = LatentContainer(
            N_samples=N_samples, N_features=N_features, dims=dims, lumped=lumped
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["model_state_dict"])

        if self.is_pub:
            # self.latents is a nn.Module with a specific parameter structure
            # Create a proper state dict with the expected parameter name
            # out of the hidden_states provided by the authors

            param_name = list(self.latents.state_dict().keys())[0]
            state_dict = {param_name: self.checkpoint["hidden_states"]}
            self.latents.load_state_dict(state_dict)
        else:
            self.latents.load_state_dict(self.checkpoint["hidden_states"])

        self.model.eval()
        self.latents.eval()

    def predict(self, coords, latent_indices, batch_size=16, normalize=True):
        """
        Generate predictions for given coordinates and latent indices.

        Args:
            coords: Coordinate tensor (spatial positions)
            latent_indices: Indices of latent codes to use
            batch_size: Batch size for inference
            normalize: Whether to apply normalization

        Returns:
            Tensor of predictions
        """
        if isinstance(latent_indices, int):
            latent_indices = [latent_indices]

        coords = (
            torch.tensor(coords, dtype=torch.float32)
            if not isinstance(coords, torch.Tensor)
            else coords
        )
        coords = coords.to(self.device)

        all_predictions = []
        latent_indices = torch.LongTensor(latent_indices)

        with torch.no_grad():
            for i in range(0, len(latent_indices), batch_size):
                batch_indices = latent_indices[i : i + batch_size].to(self.device)
                batch_latents = self.latents(batch_indices)

                if normalize and self.x_normalizer is not None:
                    normalized_coords = self.x_normalizer.normalize(coords)
                    raw_predictions = self.model(normalized_coords, batch_latents)
                    predictions = self.y_normalizer.denormalize(raw_predictions)
                else:
                    predictions = self.model(coords, batch_latents)

                all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions, dim=0)

    def get_all_predictions(self, coords, batch_size=16, normalize=True):
        """Generate predictions for all available latent codes."""
        latent_indices = torch.arange(self.latents.latents.shape[0])
        return self.predict(coords, latent_indices, batch_size, normalize)

    def create_coordinates_grid(self, shape=None):
        """
        Create a normalized coordinate grid based on data shape.

        Args:
            shape: Optional tuple of dimensions (height, width) or (depth, height, width)
                  If None, inferred from self.data

        Returns:
            Tensor of coordinates
        """
        if shape is None:
            # Infer shape from data following the pattern in cnf.scripts.train.py
            spatial_shape = (
                self.data.shape[1:-1]
                if len(self.data.shape) > 3
                else self.data.shape[1:]
            )
            coord = [np.linspace(0, 1, i) for i in spatial_shape]
            coords = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)
            return torch.tensor(coords, dtype=torch.float32)
        else:
            if len(shape) == 2:
                h, w = shape
                y_coords = torch.linspace(0, 1, h)
                x_coords = torch.linspace(0, 1, w)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
                return torch.stack([grid_y, grid_x], dim=-1)
            elif len(shape) == 3:
                d, h, w = shape
                z_coords = torch.linspace(0, 1, d)
                y_coords = torch.linspace(0, 1, h)
                x_coords = torch.linspace(0, 1, w)
                grid_z, grid_y, grid_x = torch.meshgrid(
                    z_coords, y_coords, x_coords, indexing="ij"
                )
                return torch.stack([grid_z, grid_y, grid_x], dim=-1)
            else:
                raise ValueError(f"Unsupported shape dimensionality: {shape}")
