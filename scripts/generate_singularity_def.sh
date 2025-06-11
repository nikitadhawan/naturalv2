#!/bin/bash

# generate_singularity_def.sh - Generate Singularity definition file with current pyproject.toml

set -e

# Parse command line arguments
DEF_FILE="${1:-singularity.def}"

echo "Generating definition file: $DEF_FILE"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in current directory"
    exit 1
fi

# Check if output file already exists
if [ -f "$DEF_FILE" ]; then
    read -p "File $DEF_FILE already exists. Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Creating definition file..."

# Write the base definition
cat > "$DEF_FILE" << EOF
Bootstrap: docker
From: ubuntu:24.04

%post
    # Update package lists
    apt-get update

    # Install essential system packages
    apt-get install -y \\
        build-essential \\
        ca-certificates \\
        curl \\
        git \\
        htop \\
        libmunge2 \\
        nano \\
        vim \\
        wget \\
        software-properties-common \\
        gnupg2 \\
        lsb-release

    # Install packages needed for your Python dependencies
    apt-get install -y \\
        libssl-dev \\
        libffi-dev \\
        libbz2-dev \\
        liblzma-dev \\
        libsqlite3-dev \\
        libreadline-dev \\
        zlib1g-dev \\
        libncurses5-dev \\
        libncursesw5-dev \\
        libgdbm-dev \\
        libnss3-dev \\
        libxml2-dev \\
        libxslt1-dev \\
        libcurl4-openssl-dev \\
        libgeos-dev \\
        libproj-dev \\
        libgdal-dev \\
        pkg-config

    # Install packages for GPU support and ML libraries
    apt-get install -y \\
        libnvidia-compute-535 \\
        libnvidia-decode-535 \\
        libnvidia-encode-535 \\
        nvidia-utils-535 \\
        libcurand10 \\
        libcusolver11

    # Install SLURM dependencies
    apt-get install -y \\
        munge \\
        libmunge-dev \\
        libpam0g-dev \\
        libmariadb-dev \\
        libhwloc-dev \\
        libjson-c-dev \\
        libdbus-1-dev \\
        libyaml-dev \\
        libhttp-parser-dev \\
        libsystemd-dev

    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:\$PATH"

    # Create SLURM-related directories
    mkdir -p /opt/slurm-23.11.9 /etc/munge /var/run/munge

    # Create workspace and projects directories
    mkdir -p /workspace /projects/natural /h/$USER/.ssh

    # Create pyproject.toml from embedded content
    cat > /workspace/pyproject.toml << 'PYPROJECT_EOF'
EOF

# Append the current pyproject.toml content
cat pyproject.toml >> "$DEF_FILE"

echo "PYPROJECT_EOF" >> "$DEF_FILE"

# Append the rest of the definition
cat >> "$DEF_FILE" << 'EOF'

    # Create and activate virtual environment
    cd /workspace
    /root/.local/bin/uv venv --python 3.11 --seed /venv
    . /venv/bin/activate

    # Install dependencies using uv
    /root/.local/bin/uv sync -n --dev --active

%environment
    export PATH="/root/.local/bin:$PATH"
    export PATH="/opt/slurm-23.11.9/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/slurm-23.11.9/lib:/opt/slurm-23.11.9/lib/slurm:$LD_LIBRARY_PATH"
    export SLURM_CONF="/opt/slurm-23.11.9/etc/slurm.conf"
    export PYTHONPATH="/venv/lib/python3.11/site-packages:$PYTHONPATH"
    export VIRTUAL_ENV="/venv"
    export PATH="/venv/bin:$PATH"

%runscript
    cd /workspace
    exec /bin/bash "$@"

%startscript
    cd /workspace
EOF

echo "Definition file generated successfully: $DEF_FILE"
echo ""
echo "To build the container, use one of these commands:"
echo ""
echo "  # With --remote (requires Sylabs account):"
echo "  singularity build --remote --sandbox /path/to/container_sandbox $DEF_FILE"
echo ""
echo "  # With --fakeroot (if supported on your system):"
echo "  singularity build --fakeroot --sandbox /path/to/container_sandbox $DEF_FILE"
echo ""
echo "  # With sudo (if you have root access):"
echo "  sudo singularity build --sandbox /path/to/container_sandbox $DEF_FILE"
echo ""
echo "To run the container:"
echo "  singularity shell --writable /path/to/container_sandbox"
