import os
import shutil

from mood.metrics.frustraR_Metrics import FrustraRMetrics


def main():
    license_key = "MODELIRANJE"
    frust = FrustraRMetrics(license_key=license_key)

    sequences = []
    with open("mood_job/001/relax/sequences.txt", "r") as f:
        for line in f:
            sequences.append(line.strip())

    df1 = frust.compute(sequences=sequences, iteration=1, folder_name="mood_job")
    print("Finish")


if __name__ == "__main__":

    if os.path.exists("mood_job/001/frustrar"):
        shutil.rmtree("mood_job/001/frustrar")
    main()
