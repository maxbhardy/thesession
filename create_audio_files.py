import sqlite3
import pathlib
import re

import music21
import pandas as pd
import numpy as np

from thesession.converter import ABCMusicConverter

def sanitize_title(title: str) -> str:
    # Replace spaces with underscores
    title = title.replace(" ", "_")

    # Remove any character that is NOT alphanumeric, underscore, hyphen, or dot
    title = re.sub(r"[^A-Za-z0-9_\-\.]", "", title)

    # Optionally, truncate length to e.g. 100 chars
    return title[:100].lower()

instruments: dict = {
    k.lower(): v
    for k, v in vars(music21.instrument).items()
    if hasattr(v, "bestName")
}

instruments = [
    "piano",
    "accordion",
    "violin",
    "acousticguitar",
    "banjo",
    "flute",
]

tempos = [i for i in range(120, 240, 20)]


query = """
SELECT TuneID, TuneVersionID, TuneTitle, TuneVersion
FROM TuneVersionView
JOIN Tunes USING (TuneID)
WHERE TuneVersionNumber < 6
ORDER BY TuneID, TuneVersionID
"""


# Fetch tunes
with sqlite3.connect("database2.db") as con:
    tunes = pd.read_sql(query, con)

con.close()


# Random generator
prng = np.random.default_rng()
num_audio = 5

# Create audio files
root = pathlib.Path("audio")
root.mkdir(exist_ok=True)

for row in tunes.iloc[:1].itertuples():
    print(row.TuneID, row.TuneTitle, row.TuneVersionID)

    # Create destination directory
    title = sanitize_title(row.TuneTitle)
    dest = root / f"{row.TuneID}_{title}"
    dest.mkdir(exist_ok=True)
    
    # Define the selected instrument and tempo for the tune
    tmp_instruments = prng.choice(instruments, num_audio, replace=False, shuffle=True)
    tmp_tempos = prng.integers(120, 240, size=num_audio)
    tmp_wraps = prng.uniform(0, 1, size=num_audio)
    tmp_noises = prng.uniform(0, 0.002, size=num_audio)

    for i, (instr, t, w, n) in enumerate(zip(tmp_instruments, tmp_tempos, tmp_wraps, tmp_noises)):
        filename = f"{row.TuneVersionID}_{i}"
        
        ABCMusicConverter(row.TuneVersion, filename, dest).to_mp3(
            instrument=instr,
            tempo=t,
            max_duration=300,
            cut_silence=30,
            wrap=w,
            noise_amplitude=n,
            vbr=8,
            clean_files=True
        )


