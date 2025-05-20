"""Utility to start a vLLM server on SLURM or locally."""

import importlib.util
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
from typing import Optional

import click
import submitit
from rich.console import Console
from rich.table import Table


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _check_vllm_installed() -> None:
    """Check if vLLM is installed."""
    if not importlib.util.find_spec("vllm"):
        raise click.UsageError(
            "vLLM is not installed. Please install it using 'pip install vllm'."
        )


def _process_env_vars(env: list[str]) -> dict[str, str]:
    """Process environment variables from the command line."""
    env_vars: dict[str, str] = {}
    if env:
        for env_var in env:
            if "=" not in env_var:
                raise click.UsageError(
                    f"Invalid format for --env argument: {env_var}. "
                    "Expected format is KEY=VALUE."
                )
            key, value = env_var.split("=", 1)
            key = key.strip()
            if not key:
                raise click.UsageError(
                    f"Invalid format for --env argument: {env_var}. "
                    "KEY cannot be empty."
                )
            env_vars[key] = value
    return env_vars


def _display_slurm_job_details(
    job: submitit.Job,
    log_folder: str,
    job_name: str,
    nodes: int,
    ntasks_per_node: int,
    cpus_per_task: int,
    mem_per_cpu_gb: int,
    gpus_per_node: Optional[int],
    partition: str,
    qos: Optional[str],
    time: str,
    constraint: Optional[str],
    exclude: Optional[str],
    setup: Optional[list[str]],
    env_vars: dict[str, str],
    args: list[str],
    console: Console,
) -> None:
    """Display details of the SLURM job."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Param", style="dim")
    table.add_column("Value")
    table.add_row("Job Name", job_name, style="bold green")
    table.add_row("Nodes", str(nodes))
    table.add_row("Tasks per Node", str(ntasks_per_node))
    table.add_row("CPUs per Task", str(cpus_per_task))
    table.add_row("Memory per CPU (GB)", str(mem_per_cpu_gb))
    table.add_row("GPUs per Node", str(gpus_per_node) or "N/A")
    table.add_row("Partition", partition or "N/A")
    table.add_row("Time Limit", time)
    table.add_row("QOS", qos or "N/A")
    table.add_row(
        "Constraint", "\n".join(constraint.split(",")) if constraint else "N/A"
    )
    table.add_row("Excluded Nodes", "\n".join(exclude.split(",")) if exclude else "N/A")
    table.add_row("Setup Commands", "\n".join(list(setup)) if setup else "N/A")
    table.add_row("\nEnvironment Variables", style="magenta")
    for key, value in env_vars.items():
        table.add_row(f"  {key}:", f"{str(value)}")
    table.add_row("\nvLLM Args", style="magenta")
    _add_vllm_args_to_table(args, table)
    table.add_row("Job ID", str(job.job_id), style="bold green")
    table.add_row(
        "Log Folder",
        log_folder.replace("%j", str(job.job_id)),
        style="bold blue",
    )
    console.print(table)


def _display_local_execution_details(
    env_vars: dict[str, str], args: list[str], console: Console
) -> None:
    """Display details of the local execution."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Param", style="dim")
    table.add_column("Value")
    table.add_row("Environment Variables", style="magenta")
    for key, value in env_vars.items():
        table.add_row(f"  {key}:", f"{str(value)}", style="bold")
    table.add_row("vLLM Args", style="magenta")
    _add_vllm_args_to_table(args, table)
    console.print(table)


def _add_vllm_args_to_table(args: list[str], table: Table) -> None:
    """Add vLLM arguments to the table."""
    i = 0
    while i < len(args):
        if args[i].startswith("--") and "=" in args[i]:
            arg, value = args[i].split("=", 1)
            table.add_row(arg, value)
            i += 1
        elif (
            i + 1 < len(args)
            and args[i].startswith("--")
            or i + 1 < len(args)
            and args[i].startswith("-")
        ):
            arg = args[i]
            value = args[i + 1]
            table.add_row(arg, value)
            i += 2


def _start_vllm_server(
    *args: str, model_name: str, env_keys_for_logging: list[str]
) -> None:
    """Start the vLLM server with the given arguments."""
    for key in env_keys_for_logging:
        logging.info(
            f"Environment variable '{key}' : '{os.environ.get(key, '<< Not Set >>')}'"
        )

    vllm_args = list(args)
    hostname = socket.gethostname()
    logging.info(f"Starting vLLM server on host: {hostname} with args: {vllm_args}")

    command = ["vllm", "serve", model_name]

    # force --host to be 0.0.0.0 to allow access from outside the node
    filtered_vllm_args: list[str] = []
    i = 0
    while i < len(vllm_args):
        arg = vllm_args[i]
        if arg == "--host":
            logging.warning(
                f"User provided '{arg} {vllm_args[i + 1]}'. "
                "Overriding with '--host 0.0.0.0' for SLURM job accessibility."
            )
            i += 2
        else:
            filtered_vllm_args.append(arg)
            i += 1

    command.extend(["--host", "0.0.0.0"])
    command.extend(filtered_vllm_args)

    logging.info(
        f"Starting vLLM server with command: {' '.join(shlex.quote(str(s)) for s in command)}"
    )

    subprocess.run(command, check=True, text=True, stderr=subprocess.STDOUT)
    logging.info("vLLM server started successfully.")


def launch_vllm_server(  # noqa: PLR0912, PLR0915
    vllm_args: list[str],
    local: bool = False,
    nodes: int = 1,
    ntasks_per_node: int = 1,
    gpus_per_node: Optional[int] = None,
    cpus_per_task: int = 1,
    mem_per_cpu_gb: int = 1,
    partition: Optional[str] = None,
    qos: Optional[str] = None,
    time: str = "00:30:00",
    constraint: Optional[str] = None,
    exclude: Optional[str] = None,
    log_folder: Optional[str] = None,
    job_name: str = "vllm_server",
    env_vars: Optional[dict[str, str]] = None,
    setup: Optional[list[str]] = None,
) -> Optional[submitit.Job]:
    """Launch a vLLM server on SLURM or locally.

    Parameters
    ----------
    vllm_args : list[str]
        List of arguments to pass to the vLLM server. Can be any valid vLLM arguments
        with one or a combination of the following formats:
        1. ["--arg", "value"]
        2. ["--arg=value"]
    local : bool, default=False
        If True, run the job locally. If False, run the job on SLURM.
    nodes : int, default=1
        Number of nodes on which to run the job. Ignored if ``local=True``.
    ntasks_per_node : int, default=1
        Number of tasks to invoke on each node. Ignored if ``local=True``.
    gpus_per_node : int, optional
        Number of GPUs required per allocated node. Ignored if ``local=True``.
    cpus_per_task : int, default=1
        Number of CPUs required per task. Ignored if ``local=True``.
    mem_per_cpu_gb : int, default=1
        Maximum amount of real memory in GB per allocated CPU required by the job.
        Ignored if ``local=True``.
    partition : str, optional
        SLURM partition requested. Required unless ``local=True``.
    qos : str, optional
        SLURM quality of service. Ignored if ``local=True``.
    time : str, default="00:30:00"
        SLURM time limit in the format HH:MM:SS. Ignored if ``local=True``.
    constraint : str, optional, default=None
        List of constraints for SLURM job submission. Ignored if ``local=True``.
    exclude : str, optional, default=None
        Comma-separated list of nodes to exclude from the job. Ignored if ``local=True``.
    log_folder : str, optional, default=None
        Folder to store logs. If None, defaults to ``"outputs/%j"``, where %j is the
        job ID. If running locally, this is ignored.
    job_name : str, default="vllm_server"
        Name of the job to be submitted to SLURM. Ignored if ``local=True``.
    env_vars : dict[str, str], optional, default=None
        Environment variables to set for the job. If None, no environment variables
        are set.
    setup : list[str], optional, default=None
        Setup commands to run before the job. Ignored if ``local=True``.

    Returns
    -------
    submitit.Job, optional
        The SLURM job object if running on SLURM, or ``None`` if running locally.
    """
    _check_vllm_installed()

    if "--model" not in vllm_args:
        raise ValueError(
            "The --model argument must be specified to start the vLLM server."
        )
    if not local and partition is None:
        raise ValueError("SLURM partition must be specified when not running locally.")

    # get vllm model from vllm_args
    model_arg_index = vllm_args.index("--model") + 1
    if model_arg_index >= len(vllm_args):
        raise ValueError("Model name not provided after --model argument.")

    vllm_model = vllm_args[model_arg_index]

    # remove --model from vllm_args
    vllm_args = vllm_args[: model_arg_index - 1] + vllm_args[model_arg_index + 1 :]

    if "--served-model-name" not in vllm_args:
        vllm_args.append("--served-model-name")
        vllm_args.append(vllm_model.split("/")[-1])

    # prepare environment variables
    launch_env_vars = os.environ.copy()
    str_env_vars = {}
    if env_vars:
        str_env_vars = {k: str(v) for k, v in env_vars.items()}
        launch_env_vars.update(str_env_vars)
        logging.debug(f"Environment variables to set/override: {env_vars}")
    env_keys_for_logging = list(str_env_vars.keys())

    if local:
        # warn if local and slurm options are set
        if (
            log_folder != "outputs/%j"
            or job_name != "vllm_server"
            or cpus_per_task != 1
            or ntasks_per_node != 1
            or nodes != 1
            or partition is not None
            or qos is not None
            or time != "00:30:00"
            or constraint is not None
            or exclude is not None
            or mem_per_cpu_gb != 1
            or gpus_per_node is not None
        ):
            logging.warning(
                "Ignoring SLURM options because --local is set. Running locally."
            )

        # warn if 'setup' is set
        if setup:
            logging.warning(
                "Ignoring setup commands because --local is set. Running locally."
            )

        command = ["vllm", "serve", vllm_model] + vllm_args
        logging.info(
            f"Running vLLM server locally with command: {' '.join(shlex.quote(str(s)) for s in command)}"
        )

        process: Optional[subprocess.Popen] = None

        # handle signals - SIGINT, SIGTERM
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, attempting graceful shutdown...")
            if process and process.poll() is None:
                logging.info(f"Sending SIGTERM to process {process.pid}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                    logging.info("vLLM server terminated gracefully.")
                except TimeoutError:
                    logging.warning(
                        "vLLM server did not terminate in time, killing it."
                    )
                    process.kill()
                    process.wait()
                    logging.info("vLLM server killed.")
                    sys.exit(128 + signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            process = subprocess.Popen(command, env=launch_env_vars)
            logging.info(
                f"vLLM server started with PID {process.pid}. Press Ctrl+C to stop."
            )
            return_code = process.wait()

            if return_code != 0:
                logging.error(f"vLLM server exited with error code {return_code}.")
            else:
                logging.info("vLLM server exited successfully.")
        except KeyboardInterrupt:
            logging.info("vLLM server interrupted by user.")
            signal_handler(signal.SIGINT, None)
            sys.exit(0)

        return None

    # NOTE: Valid arguments: nodes, constraint, time, job_name, mem_per_cpu, setup,
    # additional_parameters, exclude, qos, gpus_per_node, dependency, mail_type,
    # exclusive, num_gpus, gpus_per_task, nodelist, cpus_per_task, ntasks_per_node,
    # cpus_per_gpu, mem_per_gpu, comment, account, wckey, signal_delay_s, mail_user,
    # mem, use_srun, srun_args, stderr_to_stdout, partition, array_parallelism, gres
    slurm_params = {
        "slurm_job_name": job_name,
        "slurm_partition": partition,
        "slurm_time": time,
        "slurm_nodes": nodes,
        "slurm_ntasks_per_node": ntasks_per_node,
        "slurm_cpus_per_task": cpus_per_task,
        "slurm_mem_per_cpu": f"{mem_per_cpu_gb}G",
        "stderr_to_stdout": True,
    }

    if qos:
        slurm_params["slurm_qos"] = qos
    if gpus_per_node:
        slurm_params["slurm_gres"] = f"gpu:{gpus_per_node}"
    if constraint:
        slurm_params["slurm_constraint"] = constraint
    if exclude:
        slurm_params["slurm_exclude"] = exclude

    setup_cmds = setup or []
    if str_env_vars:
        for key, value in str_env_vars.items():
            setup_cmds.append(f"export {key}={shlex.quote(value)}")

    if setup_cmds:
        slurm_params["slurm_setup"] = setup_cmds
        logging.debug(f"Setup commands to run before the job: {setup}")
    logging.debug(f"Submitting job to SLURM with parameters: {slurm_params}")

    if log_folder is None:
        log_folder = "outputs/%j"

    log_dir = log_folder
    if "%j" in log_folder:
        log_dir = os.path.dirname(log_folder)
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(**slurm_params)
    return executor.submit(
        _start_vllm_server,
        *vllm_args,
        model_name=vllm_model,
        env_keys_for_logging=env_keys_for_logging,
    )


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": ["-h", "--help"],
    },
    help="Launch a vLLM OpenAI-compatible server, either locally or on SLURM. "
    "Pass execution/SLURM options first, followed by any valid vLLM arguments. ",
)
@click.option("--local", is_flag=True, help="run the job locally")
@click.option(
    "--log-folder",
    type=str,
    default="outputs/%j",
    show_default=True,
    help="folder to store logs",
)
@click.option(
    "-c",
    "--cpus-per-task",
    type=int,
    default=1,
    show_default=True,
    help="number of CPUs required per task",
)
@click.option(
    "-j",
    "--job-name",
    type=str,
    default="vllm_server",
    show_default=True,
    help="name of the job",
)
@click.option(
    "--ntasks-per-node",
    type=int,
    default=1,
    show_default=True,
    help="number of tasks to invoke on each node; ignored if --local is set",
)
@click.option(
    "-n",
    "--nodes",
    type=int,
    default=1,
    show_default=True,
    help="number of nodes on which to run the job; ignored if --local is set",
)
@click.option(
    "-p",
    "--partition",
    type=str,
    help="SLURM partition requested; required unless --local is set",
)
@click.option(
    "-q", "--qos", type=str, help="SLURM quality of service; ignored if --local is set"
)
@click.option(
    "-t",
    "--time",
    type=str,
    default="00:30:00",
    show_default=True,
    help="SLURM time limit in the format HH:MM:SS; ignored if --local is set",
)
@click.option(
    "-C",
    "--constraint",
    type=str,
    help="Constraints for SLURM job submission; ignored if --local is set",
)
@click.option(
    "--exclude",
    type=str,
    help="Comma-separated list of nodes to exclude from the job; ignored if --local is set",
)
@click.option(
    "--mem-per-cpu-gb",
    type=int,
    default=1,
    show_default=True,
    help="maximum amount of real memory in GB per allocated cpu required by the job; ignored if --local is set",
)
@click.option(
    "--gpus-per-node",
    type=int,
    help="number of GPUs required per allocated node; ignored if --local is set",
)
@click.option(
    "--env",
    multiple=True,
    type=str,
    help="environment variables to set for the job; format: KEY=VALUE; can be repeated to set multiple variables",
)
@click.option(
    "--setup",
    multiple=True,
    type=str,
    help="setup commands to run before the job; can be repeated to set multiple commands; ignored if --local is set",
)
@click.pass_context
def cli_main(
    ctx: click.Context,
    local: bool,
    log_folder: str,
    cpus_per_task: int,
    job_name: str,
    ntasks_per_node: int,
    nodes: int,
    partition: str,
    qos: str,
    time: str,
    constraint: str,
    exclude: str,
    mem_per_cpu_gb: int,
    gpus_per_node: int,
    env: list[str],
    setup: list[str],
):
    """Start an OpenAI-compatible vLLM server on a SLURM cluster or locally."""
    env_vars = _process_env_vars(env)
    console = Console()

    try:
        job = launch_vllm_server(
            vllm_args=list(ctx.args),
            local=local,
            log_folder=log_folder,
            cpus_per_task=cpus_per_task,
            job_name=job_name,
            ntasks_per_node=ntasks_per_node,
            nodes=nodes,
            partition=partition,
            qos=qos,
            time=time,
            constraint=constraint,
            exclude=exclude,
            mem_per_cpu_gb=mem_per_cpu_gb,
            gpus_per_node=gpus_per_node,
            env_vars=env_vars,
            setup=list(setup) if setup else None,
        )

        if job:
            _display_slurm_job_details(
                job,
                log_folder,
                job_name,
                nodes,
                ntasks_per_node,
                cpus_per_task,
                mem_per_cpu_gb,
                gpus_per_node,
                partition,
                qos,
                time,
                constraint,
                exclude,
                setup,
                env_vars,
                ctx.args,
                console,
            )
        elif local:
            _display_local_execution_details(env_vars, ctx.args, console)
        else:
            click.echo(
                "Launch process completed without submitting a SLURM job or running locally.",
                color=True,
            )
    except (ValueError, FileNotFoundError, click.UsageError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
