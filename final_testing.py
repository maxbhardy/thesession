import os
import dotenv
import pathlib

import numpy as np
import pandas as pd

from mpi4py import MPI

from thesession.retriever import TheSessionRetriever, get_database_url

dotenv.load_dotenv("thesession-db/.env")

url = get_database_url(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    port=15432,
    database=os.getenv("POSTGRES_DB"),
)

retriever = TheSessionRetriever(
    url, "models/step6_best.pt", backend="soundfile", device="cpu"
)

# Dataset
dataset = pd.read_csv("dataset.csv")

test_subset = dataset.loc[dataset["dataset"] == "test", "tune"]

# Communication MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# Select subset according to mpi rank
indices = np.array_split(np.arange(len(test_subset)), mpi_size)[mpi_rank]
test_subset = test_subset.iloc[indices].copy()

######################################################################
# Final testing
correct_tune_top1 = 0
correct_tune_top5 = 0
correct_tune_top10 = 0
correct_version_top1 = 0
correct_version_top5 = 0
correct_version_top10 = 0

denom = 0

root = pathlib.Path("audio_flac")

for i, tune in enumerate(test_subset):
    # Fetch tune id from directory
    tune_id = int(tune.split("_")[0])

    # Iterate over files in directory
    if (root / tune).exists():
        for file in (root / tune).iterdir():
            if file.exists():
                tune_version_id = int(file.stem.split("_")[0])

                # Get results from retriever
                results = retriever(file, limit=10, duration=60)

                # Add to denominator
                denom += 1

                # Check if the correct tune is identified
                tune_results = results["TuneID"].to_list()

                if tune_id in tune_results[:1]:
                    correct_tune_top1 += 1
                    correct_tune_top5 += 1
                    correct_tune_top10 += 1
                elif tune_id in tune_results[1:5]:
                    correct_tune_top5 += 1
                    correct_tune_top10 += 1
                elif tune_id in tune_results[5:10]:
                    correct_tune_top10 += 1

                # Check if the correct tune version is identified
                version_results = results["TuneVersionID"].to_list()

                if tune_version_id in version_results[:1]:
                    correct_version_top1 += 1
                    correct_version_top5 += 1
                    correct_version_top10 += 1
                elif tune_version_id in version_results[1:5]:
                    correct_version_top5 += 1
                    correct_version_top10 += 1
                elif tune_version_id in version_results[5:10]:
                    correct_version_top10 += 1

###############################################################
# Only root process do the stats
correct_tune_top1 = mpi_comm.gather(correct_tune_top1, root=0)
correct_tune_top5 = mpi_comm.gather(correct_tune_top5, root=0)
correct_tune_top10 = mpi_comm.gather(correct_tune_top10, root=0)
correct_version_top1 = mpi_comm.gather(correct_version_top1, root=0)
correct_version_top5 = mpi_comm.gather(correct_version_top5, root=0)
correct_version_top10 = mpi_comm.gather(correct_version_top10, root=0)
denom = mpi_comm.gather(denom, root=0)

if mpi_rank == 0:
    total_denom = sum(denom)
    tune_acc_top1 = sum(correct_tune_top1) / total_denom
    tune_acc_top5 = sum(correct_tune_top5) / total_denom
    tune_acc_top10 = sum(correct_tune_top10) / total_denom
    version_acc_top1 = sum(correct_version_top1) / total_denom
    version_acc_top5 = sum(correct_version_top5) / total_denom
    version_acc_top10 = sum(correct_version_top10) / total_denom

    accuracy = pd.DataFrame(
        {
            "TuneAccuracy": [tune_acc_top1, tune_acc_top5, tune_acc_top10],
            "VersionAccuracy": [version_acc_top1, version_acc_top5, version_acc_top10],
        },
        index=["top1", "top5", "top10"],
    )

    accuracy.to_csv("test/final_test.csv")
