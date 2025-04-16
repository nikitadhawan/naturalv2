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


def start_vllm_server(
    *args: tuple[str], model_name: str, env_keys_for_logging: list[str]
):
    """Start the vLLM server with the given arguments."""
    for key in env_keys_for_logging:
        logging.info(
            f"Environment variable '{key}' : {os.environ[key], '<< Not Set >>'}"
        )

    vllm_args = list(args)
    hostname = socket.gethostname()
    logging.info(f"Starting vLLM server on host: {hostname} with args: {vllm_args}")

    command = ["vllm", "serve", model_name]

    # force --host to be 0.0.0.0 to allow access from outside the node
    filtered_vllm_args = []
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
    command.extend(vllm_args)

    logging.info(
        f"Starting vLLM server with command: {' '.join(shlex.quote(str(s)) for s in command)}"
    )

    try:
        process = subprocess.run(
            command, check=True, text=True, stderr=subprocess.STDOUT
        )
        logging.info(
            f"vLLM server started with PID {process.pid}. Press Ctrl+C to stop."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"vLLM Server process failed with exit code {e.returncode} ---")
        logging.info("--- [SLURM Job] Output ---:")
        output = e.stdout or e.stderr or "(No output captured)"
        print(output)
        raise
    except Exception as e:
        logging.error(
            f"[SLURM Job] An unexpected error occurred trying to run vLLM: {e}"
        )
        raise


def launch_vllm_server(  # noqa: PLR0912, PLR0915
    vllm_args: list[str],
    local: bool = False,
    log_folder: Optional[str] = None,
    cpus_per_task: int = 1,
    job_name: str = "vllm_server",
    ntasks_per_node: int = 1,
    nodes: int = 1,
    partition: Optional[str] = None,
    qos: Optional[str] = None,
    time: str = "00:30:00",
    constraint: Optional[str] = None,
    mem_per_cpu_gb: int = 1,
    gpus_per_node: Optional[int] = None,
    env_vars: Optional[dict[str, str]] = None,
    setup: Optional[list[str]] = None,
) -> Optional[submitit.Job]:
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
        logging.info(f"Environment variables to set/override: {env_vars}")
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
        except FileNotFoundError:
            logging.error("vllm command not found. Please ensure vLLM is installed.")
            raise
        except KeyboardInterrupt:
            logging.info("vLLM server interrupted by user.")
            signal_handler(signal.SIGINT, None)
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error starting vLLM server: {e}")
            raise

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

    setup_cmds = setup or []
    if str_env_vars:
        for key, value in str_env_vars.items():
            setup_cmds.append(f"export {key}={shlex.quote(value)}")

    if setup_cmds:
        slurm_params["slurm_setup"] = setup_cmds
        logging.info(f"Setup commands to run before the job: {setup}")

    logging.info(f"Submitting job to SLURM with parameters: {slurm_params}")

    if log_folder is None:
        log_folder = "outputs/%j"

    log_dir = log_folder
    if "%j" in log_folder:
        log_dir = os.path.dirname(log_folder)
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(**slurm_params)
    job = executor.submit(
        start_vllm_server,
        *vllm_args,
        model_name=vllm_model,
        env_keys_for_logging=env_keys_for_logging,
    )
    logging.info(
        f"Job submitted with ID {job.job_id}. Logs will be saved in {log_dir}."
    )
    return job


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": ["-h", "--help"],
    },
    help="Launch a vLLM OpenAI-compatible server, either locally or on SLURM. "
    "Pass execution/SLURM options first, then any valid vLLM options.",
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
    help="list of constraints for SLURM job submission; ignored if --local is set",
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
    help="environment variables to set for the job",
)
@click.option(
    "--setup",
    multiple=True,
    type=str,
    help="setup commands to run before the job; ignored if --local is set",
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
    constraint: list,
    mem_per_cpu_gb: int,
    gpus_per_node: int,
    env: list[str],
    setup: list[str],
):
    """Command line interface for starting an OpenAI-compatible vLLM server."""
    vllm_args_list = list(ctx.args)  # passthrough args

    # process environment variables
    env_vars = {}
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

    # run job and catch errors
    try:
        job = launch_vllm_server(
            vllm_args=vllm_args_list,
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
            mem_per_cpu_gb=mem_per_cpu_gb,
            gpus_per_node=gpus_per_node,
            env_vars=env_vars,
            setup=list(setup) if setup else None,
        )
        if job:  # SLURM submission successful
            click.echo("CLI SLURM Job Submission Successful", color="green")
            click.echo(f"Job ID: {job.job_id}", color="green")
            effective_log_path = log_folder.replace("%j", job.job_id)
            click.echo(f"SLURM Log Dir: {effective_log_path}", color="green")
            click.echo(f"Monitor: squeue -j {job.job_id}", color="yellow")
            click.echo(f"Cancel SLURM Job: scancel {job.job_id}", color="red")
        elif local:  # local execution started and finished (or was interrupted)
            click.echo("Local vLLM Server Process Ended", color="yellow")
        else:
            click.echo(
                "Launch process completed without submitting a SLURM job or running locally.",
                color="yellow",
            )
    except (ValueError, FileNotFoundError, click.UsageError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)  # ensure cleanup on error too
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
