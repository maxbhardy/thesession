import pathlib
import torch
import librosa

from laion_clap import CLAP_Module
from laion_clap.clap_module.htsat import create_htsat_model
from laion_clap.training.data import get_audio_features


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



class AudioConfig:
    model_type: str = "HTSAT"
    model_name: str = "tiny"
    sample_rate: int = 48000
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 480
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000

    def asdict(self):
        return {k: getattr(self, k) for k in self.__annotations__}


class CLAPAudioModel(torch.nn.Module):
    def __init__(self, enable_fusion: bool = True, device: str | None = None):
        super().__init__()

        self.audio_cfg = AudioConfig()
        self.enable_fusion = enable_fusion
        self.device = device

        # Model audio branch
        self.audio_branch = create_htsat_model(self.audio_cfg, self.enable_fusion, "aff_2d")

        # Model audio projection
        self.audio_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

        # Send model to device
        if self.device:
            self.audio_branch = self.audio_branch.to(self.device)
            self.audio_projection = self.audio_projection.to(self.device)

    def get_audio_features(self, audio_waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        audio_features = get_audio_features(
            {},
            audio_waveform,
            480000,
            data_truncating='fusion' if self.enable_fusion else 'rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=self.audio_cfg.asdict(),
            require_grad=audio_waveform.requires_grad
        )

        return audio_features

    def get_audio_embedding_from_filelist(self, x: list[str | pathlib.Path]) -> torch.Tensor:
        """get audio embeddings from the audio file list

        Parameters
        ----------
        x: List[str] (N,): 
            an audio file list to extract features, audio files can have different
            lengths (as we have the feature fusion machanism)

        Returns
        ----------
        audio_embed : torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        audio_input = []

        for f in x:
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, _ = librosa.load(f, sr=48000)           
            audio_waveform = torch.from_numpy(audio_waveform).float()

            tmp_dict = self.get_audio_features(audio_waveform)

            audio_input.append(tmp_dict)

        audio_embed = self.get_audio_embedding(audio_input)

        return audio_embed

    def get_audio_embedding_from_data(self, x: torch.Tensor) -> torch.Tensor:
        """get audio embeddings from the audio data

        Parameters
        ----------
        x: torch.Tensor (N,T): 
            audio data, must be mono audio tracks.
        
        Returns
        ----------
        audio embed: torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        audio_input = []

        for audio_waveform in x:          
            tmp_dict = self.get_audio_features(audio_waveform)
            audio_input.append(tmp_dict)

        audio_embed = self.get_audio_embedding(audio_input)

        return audio_embed

    def get_audio_embedding(self, data: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        input_dict = {}
        keys = data[0].keys()

        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(self.device)

        return self.forward(input_dict)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.audio_branch(x, mixup_lambda=None, device=self.device)["embedding"]
        x = self.audio_projection(x)
        x = torch.nn.functional.normalize(x, dim=-1)

        return x

