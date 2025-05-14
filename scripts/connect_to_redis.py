import os
import subprocess
from typing import Optional

import click
import submitit
from submitit.helpers import CommandFunction


@click.command()
@click.option(
    "--bastion-key",
    type=str,
    required=True,
    help="Path to the bastion key .pem file.",
)
@click.option(
    "--redis-host",
    type=str,
    required=True,
    help="URL of the Redis server.",
)
@click.option(
    "--redis-port-remote",
    type=int,
    default=6379,
    help="Port of the Redis server (remote).",
)
@click.option(
    "--redis-port-local",
    type=int,
    default=6379,
    help="Port of the Redis server (local).",
)
@click.option(
    "--remote-gateway-username",
    type=str,
    required=True,
    help="Username for the remote server.",
)
@click.option(
    "--remote-gateway-hostname",
    type=str,
    required=True,
    help="Hostname or IP address of the remote gateway to the Redis server.",
)
@click.option(
    "--slurm-job-name",
    type=str,
    default="redis-gateway",
    help="SLURM job name.",
)
@click.option(
    "--slurm-partition",
    type=str,
    required=False,
    help="SLURM partition name.",
)
@click.option(
    "--slurm-qos",
    type=str,
    required=False,
    help="SLURM Quality of Service.",
)
@click.option(
    "--slurm-time",
    type=str,
    required=False,
    help="SLURM time limit.",
)
@click.option(
    "--server-alive-interval",
    type=int,
    default=60,  # Send keepalive every 60 seconds
    help="SSH ServerAliveInterval in seconds.",
)
@click.option(
    "--server-alive-count-max",
    type=int,
    default=3,  # Allow 3 unresponsive keepalives before disconnecting
    help="SSH ServerAliveCountMax.",
)
def cli_main(
    bastion_key: str,
    redis_host: str,
    redis_port_remote: int,
    redis_port_local: int,
    remote_gateway_username: str,
    remote_gateway_hostname: str,
    slurm_job_name: str,
    slurm_partition: Optional[str],
    slurm_qos: Optional[str],
    slurm_time: Optional[str],
    server_alive_interval: int,
    server_alive_count_max: int,
):
    if not os.path.exists(bastion_key):
        raise ValueError(f"Key file {bastion_key} does not exist.")

    is_local = (
        slurm_job_name == "redis-gateway"
        and slurm_partition is None
        and slurm_qos is None
        and slurm_time is None
    )

    ssh_command = [
        "ssh",
        "-o",
        f"ServerAliveInterval={server_alive_interval}",
        "-o",
        f"ServerAliveCountMax={server_alive_count_max}",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "ExitOnForwardFailure=yes",  # Exit if port forwarding setup fails
        "-i",
        bastion_key.strip(),
        "-N",
        "-L",
        f"0.0.0.0:{redis_port_local}:{redis_host.strip()}:{redis_port_remote}",
        f"{remote_gateway_username.strip()}@{remote_gateway_hostname.strip()}",
    ]

    if is_local:
        try:
            process = subprocess.Popen(
                ssh_command,
                stdout=subprocess.DEVNULL,  # redirect stdout to /dev/null
                stderr=subprocess.DEVNULL,  # redirect stderr to /dev/null
                start_new_session=True,  # for detachment, running in the background
            )
            print(
                f"SSH tunnel process started locally in new session with PID: {process.pid}. "
                "It should continue running after this script exits."
            )
        except Exception as e:
            click.echo(f"Failed to start local SSH process: {e}", err=True)
    else:
        print("Submitting SLURM job...")
        executor = submitit.AutoExecutor(folder="outputs/%j")
        executor.update_parameters(
            name=slurm_job_name,
            slurm_partition=slurm_partition,
            slurm_qos=slurm_qos,
            slurm_time=slurm_time,
            slurm_nodes=1,
            slurm_ntasks_per_node=1,
            slurm_cpus_per_task=1,
            slurm_mem_per_cpu="200M",
            stderr_to_stdout=True,
            slurm_signal_delay_s=120,
        )

        job = executor.submit(CommandFunction(ssh_command, verbose=True))
        print(f"Job submitted with ID: {job.job_id}")
        print(f"Check SLURM output in: outputs/{job.job_id}/{job.job_id}_0_log.out")
        print(f"To cancel the job: scancel {job.job_id}")


if __name__ == "__main__":
    cli_main()
