import subprocess

cpus = 50
folder_name = "/gpfs/projects/bsc72/acanella/Repos/multiObjectiveOptimizationDesign/mood/metrics/scripts"
params_folder = "/gpfs/projects/bsc72/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/hmfo/params"
seed = 1235
output_folder = "/gpfs/projects/bsc72/acanella/Repos/multiObjectiveOptimizationDesign/tests/metrics/out_relax"
native_pdb = "/gpfs/projects/bsc72/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/hmfo/AF-A0A9E4RQT2@HFFCA@146.pdb"
# distances = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/distances.json"
distances = None
sequences_file = "/gpfs/projects/bsc72/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/hmfo/sequences.txt"
# cst_file = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/Felip9/FeLip9-PET-1_CA.cst"
cst_file = None
ligand_chain = "L"



try:
    cmd = f"mpirun -np {cpus} "
    cmd += f"python {folder_name}/mpi_rosetta_metrics.py "
    cmd += f"--seed {seed} "
    cmd += f"--output_folder {output_folder} "
    cmd += f"--sequences_file {sequences_file} "
    cmd += f"--params_folder {params_folder} "
    cmd += f"--native_pdb {native_pdb} "
    cmd += f"--distances {distances} "
    cmd += f"--cst_file {cst_file} "
    cmd += f"--ligand_chain {ligand_chain}"
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    stdout, stderr = proc.communicate()
    print("Output:", stdout.decode())
    print("Error:", stderr.decode())
    with open(f"{output_folder}/rosetta.out", "w") as f:
        f.write(stdout.decode())
    with open(f"{output_folder}/rosetta.out", "w") as f:
        f.write(stdout.decode())
except Exception as e:
    raise Exception(f"An error occurred while running the Rosetta metrics: {e}")