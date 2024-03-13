import io

from pkg_resources import Requirement, resource_stream


def _parallel(jobs, cpus=None, script_name="commands", folder="."):
    """
    Generates scripts to run jobs simultaneously in N Cpus in a local computer,
    i.e., without a job manager. The input jobs must be a list representing each
    job to execute as a string and the list order will be prioritized upon execution.

    Two different scripts are written to execute the jobs in bash language. For
    example, if the script_name variable is set to commands and the cpus to 4, five
    scripts will be written:

    - commands
    - commands_0
    - commands_1
    - commands_2
    - commands_3
    ...

    The jobs to execute are distributed into the numbered scripts. Each numbered
    script contains a sub set of jobs that will be executed in a sequential manner.
    The numberless script execute all the numbered scripts in the background, using
    the nohup command, and redirecting the output to different files for each numbered
    script. To execute the jobs is only necessary to execute:

    'bash commands'

    Parameters
    ----------
    jobs : list
        List of strings containing the commands to execute jobs.
    cpus : int
        Number of CPUs to use in the execution.
    script_name : str
        Name of the output scripts to execute the jobs.
    """
    # Write parallel execution scheme #

    if cpus == None:
        cpus = min([len(jobs), 10])
        print(f"Number of CPU not given, using {cpus} by default.")

    if len(jobs) < cpus:
        print("The number of jobs is less than the number of CPU.")
        cpu = len(jobs)
        print("Using %s CPU" % cpu)

    # Open script files
    zf = len(str(cpus))
    scripts = {}
    for c in range(cpus):
        scripts[c] = open(folder + "/" + script_name + "_" + str(c).zfill(zf), "w")
        scripts[c].write("#!/bin/sh\n")

    # Write jobs with list-order prioritization
    for i in range(len(jobs)):
        scripts[i % cpus].write(jobs[i])

    # Close script files
    for c in range(cpus):
        scripts[c].close()

    # Write script to execute them all in background
    with open(folder + "/" + script_name, "w") as sf:
        sf.write("#!/bin/sh\n")
        sf.write(
            "for script in "
            + folder
            + "/"
            + script_name
            + "_"
            + "?" * zf
            + "; do nohup bash $script &> ${script%.*}.nohup& done\n"
        )


# TODO see if there is a need to copy the file
def _copy_script_file(
    output_folder, script_name, no_py=False, subfolder=None, hidden=True
):
    """
    Copy a script file from the multistate_design package.

    Parameters
    ==========

    """
    # Get script
    path = "multistate_design/scripts"
    if subfolder is not None:
        path = path + "/" + subfolder

    script_file = resource_stream(
        Requirement.parse("multistate_design"), path + "/" + script_name
    )
    script_file = io.TextIOWrapper(script_file)

    # Write control script to output folder
    if no_py is True:
        script_name = script_name.replace(".py", "")

    if hidden:
        output_path = output_folder + "/._" + script_name
    else:
        output_path = output_folder + "/" + script_name

    with open(output_path, "w") as sof:  # pylint: disable=unspecified-encoding
        for l in script_file:
            sof.write(l)
