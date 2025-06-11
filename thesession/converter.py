import music21
import librosa
import soundfile
import pathlib
import subprocess

import numpy as np


class ABCMusicConverter:
    score: music21.stream.base.Score | None
    destination: pathlib.Path
    filename: str
    midi_file: pathlib.Path | None
    wav_file: pathlib.Path | None
    mp3_file: pathlib.Path | None
    flac_file: pathlib.Path | None

    instruments: dict = {
        k.lower(): v
        for k, v in vars(music21.instrument).items()
        if hasattr(v, "bestName")
    }

    def __init__(
        self,
        abc: str,
        filename: str,
        destination: str | pathlib.Path = ".",
        prng: np.random.Generator | None = None,
    ):
        self.destination = pathlib.Path(destination)
        self.filename = filename

        self.midi_file = None
        self.wav_file = None
        self.mp3_file = None

        self.score = music21.converter.parse(abc)

        if prng:
            self.prng = prng
        else:
            self.prng = np.random.default_rng()

    def to_midi(
        self,
        midi_file: str | pathlib.Path | None = None,
        instrument: str | None = None,
        tempo: int | None = None,
        max_notes: (
            int | None
        ) = None,  # If bigger than this number of notes (Cooleys = 70), do not export
    ) -> pathlib.Path:
        # Path to new midi file
        if midi_file is None:
            self.midi_file = (self.destination / self.filename).with_suffix(".mid")
        else:
            self.midi_file = pathlib.Path(midi_file)

        # Delete midi file if exists
        if self.midi_file.exists():
            self.midi_file.unlink()

        if instrument is not None:
            instrument = self.instruments.get(instrument.lower())
            for p in self.score.parts:
                p.insert(0, instrument())

        if tempo is not None:
            self.score.insert(0, music21.tempo.MetronomeMark(number=tempo))

        # Convert to midi
        if max_notes and self.score.highestTime > max_notes:
            self.midi_file = None
            return None
        else:
            mf = music21.midi.translate.music21ObjectToMidiFile(self.score)
            mf.open(self.midi_file, "wb")
            mf.write()
            mf.close()

            return self.midi_file

    def to_wav(
        self,
        wav_file: str | pathlib.Path | None = None,
        sound_font: str | pathlib.Path = "GeneralUser-GS.sf2",
        sampling_rate: int = 48000,
        cut_silence: int | None = None,
        noise_amplitude: float | None = None,
        start: float | None = None,
        duration: float | None = None,
        clean_files: bool = False,
        **kwargs,
    ) -> pathlib.Path:
        # Create midi file if necessary
        if self.midi_file is None:
            self.to_midi(**kwargs)

        # Stop if self.midi_file is still none (error)
        if self.midi_file is None:
            return None

        # Path to new wav file
        if wav_file is None:
            self.wav_file = (self.destination / self.filename).with_suffix(".wav")
        else:
            self.wav_file = pathlib.Path(wav_file)

        # Remove file if exists
        if self.wav_file.exists():
            self.wav_file.unlink()

        # Check if sound_font exists
        sound_font = pathlib.Path(sound_font)
        assert sound_font.exists()

        # Convert to wav
        command = [
            "fluidsynth",
            "-ni",
            str(sound_font),
            str(self.midi_file),
            "-F",
            str(self.wav_file),
            "-r",
            str(sampling_rate),
        ]

        subprocess.run(command, check=True, capture_output=True)

        if cut_silence or start or duration or noise_amplitude:
            signal, sr = librosa.load(self.wav_file, sr=sampling_rate)

        # cut silence
        if cut_silence:
            signal, _ = librosa.effects.trim(signal, top_db=cut_silence)

        # Wrap around itself
        if start:
            length = len(signal)
            new_start = int(start * length)
            signal = np.concatenate([signal, signal])[new_start : new_start + length]

        # Repeat until duration
        if duration:
            length = len(signal)
            expected_length = int(np.round(duration * sampling_rate))
            repeats = int(np.ceil(expected_length / length))
            signal = np.tile(signal, repeats)[:expected_length]

        # Add noise
        if noise_amplitude:
            noise = self.prng.normal(0, noise_amplitude, signal.shape[0])
            signal = signal + noise

        # Write new file
        if cut_silence or start or duration or noise_amplitude:
            soundfile.write(self.wav_file, signal, sampling_rate)

        if clean_files and self.midi_file.exists():
            self.midi_file.unlink()

        return self.wav_file

    def to_mp3(
        self,
        mp3_file: str | pathlib.Path | None = None,
        vbr: int | None = None,
        cbr: int | None = None,
        clean_files: bool = False,
        **kwargs,
    ) -> pathlib.Path:
        # Create wave file if necessary
        if self.wav_file is None:
            self.to_wav(**kwargs)

        # Stop if self.wav_file is still none (error)
        if self.wav_file is None:
            return None

        # Path to new mp3 file
        if mp3_file is None:
            self.mp3_file = (self.destination / self.filename).with_suffix(".mp3")
        else:
            self.mp3_file = pathlib.Path(mp3_file)

        # Remove file if exists
        if self.mp3_file.exists():
            self.mp3_file.unlink()

        # Define command
        command = ["ffmpeg", "-i", str(self.wav_file)]

        # Adjust bit rate
        if vbr:
            command += ["-q:a", f"{vbr}"]
        elif cbr:
            command += ["-b:a", f"{cbr}k"]

        command.append(str(self.mp3_file))

        subprocess.run(command, check=True, capture_output=True)

        if clean_files and self.midi_file.exists():
            self.midi_file.unlink()

        if clean_files and self.wav_file.exists():
            self.wav_file.unlink()

        return self.mp3_file
    
    def to_flac(
        self,
        flac_file: str | pathlib.Path | None = None,
        sampling_rate: int = 48000,
        audio_channels: int | None = None,
        clean_files: bool = False,
        **kwargs,
    ) -> pathlib.Path:
        # Create wave file if necessary
        if self.wav_file is None:
            self.to_wav(sampling_rate=sampling_rate, **kwargs)

        # Stop if self.wav_file is still none (error)
        if self.wav_file is None:
            return None

        # Path to new mp3 file
        if flac_file is None:
            self.flac_file = (self.destination / self.filename).with_suffix(".flac")
        else:
            self.flac_file = pathlib.Path(flac_file)

        # Remove file if exists
        if self.flac_file.exists():
            self.flac_file.unlink()

        # Define command
        command = ["ffmpeg", "-i", str(self.wav_file), "-ar", str(sampling_rate)]

        # Write audio channels
        if audio_channels:
            command += ["-ac", str(audio_channels)]

        command.append(str(self.flac_file))

        subprocess.run(command, check=True, capture_output=True)

        if clean_files and self.midi_file.exists():
            self.midi_file.unlink()

        if clean_files and self.wav_file.exists():
            self.wav_file.unlink()

        return self.flac_file
    