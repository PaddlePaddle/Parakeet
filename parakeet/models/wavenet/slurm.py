# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility module for restarting training when using SLURM.
"""
import subprocess
import os
import sys
import shlex
import re
import time


def job_info():
    """Get information about the current job using `scontrol show job`.
    Returns a dict mapping parameter names (e.g. "UserId", "RunTime", etc) to
    their values, both as strings.
    """
    job_id = int(os.environ["SLURM_JOB_ID"])

    command = ["scontrol", "show", "job", str(job_id)]
    output = subprocess.check_output(command).decode("utf-8")

    # Use a regex to extract the parameter names and values
    pattern = "([A-Za-z/]*)=([^ \t\n]*)"
    return dict(re.findall(pattern, output))


def parse_hours(text):
    """Parse a time format HH or DD-HH into a number of hours."""
    hour_chunks = text.split("-")
    if len(hour_chunks) == 1:
        return int(hour_chunks[0])
    elif len(hour_chunks) == 2:
        return 24 * int(hour_chunks[0]) + int(hour_chunks[1])
    else:
        raise ValueError("Unexpected hour format (expected HH or "
                         "DD-HH, but got {}).".format(text))


def parse_time(text):
    """Convert slurm time to an integer.
    Expects time to be of the form:
    "hours:minutes:seconds" or "day-hours:minutes:seconds".
    """
    hours, minutes, seconds = text.split(":")
    try:
        return parse_hours(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except ValueError as e:
        raise ValueError("Error parsing time {}. Got error {}.".format(text,
                                                                       str(e)))


def restart_command():
    """Using the environment and SLURM command, create a command that, when,
    run, will enqueue a repeat of the current job using `sbatch`.
    Return the command as a list of strings, suitable for passing to
    `subprocess.check_call` or similar functions.
    Returns:
        resume_command: list<str>, command to run to restart job.
        end_time: int or None; the time the job will end or None
            if the job has unlimited runtime.
    """
    # Make sure `RunTime` could be parsed correctly.
    while job_info()["RunTime"] == "INVALID":
        time.sleep(1)

    # Get all the necessary information by querying SLURM with this job id
    info = job_info()

    try:
        num_cpus = int(info["CPUs/Task"])
    except KeyError:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])

    num_tasks = int(os.environ["SLURM_NTASKS"])
    nodes = info["NumNodes"]
    gres, partition = info.get("Gres"), info.get("Partition")
    stderr, stdout = info.get("StdErr"), info.get("StdOut")
    job_name = info.get("JobName")
    command = [
        "sbatch", "--job-name={}".format(job_name),
        "--ntasks={}".format(num_tasks)
    ]

    if partition:
        command.extend(["--partition", partition])

    if gres and gres != "(null)":
        command.extend(["--gres", gres])
        num_gpu = int(gres.split(':')[-1])
        print("number of gpu assigned by slurm is {}".format(num_gpu))

    if stderr:
        command.extend(["--error", stderr])

    if stdout:
        command.extend(["--output", stdout])

    python = subprocess.check_output(
        ["/usr/bin/which", "python3"]).decode("utf-8").strip()
    dist_setting = ['-m', 'paddle.distributed.launch']
    wrap_cmd = ["srun", python, '-u'] + dist_setting + sys.argv

    command.append("--wrap={}".format(" ".join(
        shlex.quote(arg) for arg in wrap_cmd)))
    time_limit_string = info["TimeLimit"]
    if time_limit_string.lower() == "unlimited":
        print(
            "UNLIMITED detected: restart OFF, infinite learning ON.",
            flush=True)
        return command, None
    time_limit = parse_time(time_limit_string)
    runtime = parse_time(info["RunTime"])
    end_time = time.time() + time_limit - runtime

    return command, end_time
