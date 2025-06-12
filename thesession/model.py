import pathlib
import torch
from laion_clap import CLAP_Module


class TheSessionModel(torch.nn.Module):
    clap_model: CLAP_Module

    def __init__(self, enable_fusion: bool = True, device: str | None = None, **kwargs):
        super().__init__()

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.clap_model = CLAP_Module(
            enable_fusion=enable_fusion, device=self.device, **kwargs
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.clap_model.get_audio_embedding_from_data(X, use_tensor=True)

    def load_ckpt(
        self, ckpt: str | None = None, model_id: int = -1, verbose: bool = True
    ):
        return self.clap_model.load_ckpt(ckpt, model_id, verbose)

    def load(self, file: str | pathlib.Path):
        self.load_state_dict(
            torch.load(file, weights_only=True, map_location=self.device)
        )

    def save(self, file: str | pathlib.Path):
        file = pathlib.Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), file)

    def toggle_gradients(
        self,
        requires_grad: bool = False,
        parameters: list[str] | None = None,
        verbose: bool = True,
    ):
        for name, param in self.named_parameters():
            if parameters:
                toggle = any(name.startswith(p) for p in parameters)
            else:
                toggle = True

            if toggle:
                if verbose:
                    print(
                        "Enabling" if requires_grad else "Disabling",
                        "gradient for parameter",
                        name,
                    )

                param.requires_grad = requires_grad
