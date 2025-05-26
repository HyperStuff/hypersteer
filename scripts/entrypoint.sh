#!/bin/bash
set -e # Exit on error

# Start a new SSH agent
eval $(ssh-agent -s)

# Get SSH key name from environment variable or use default
SSH_KEY_NAME=${SSH_KEY_NAME:-github}
# Add the SSH key if it exists
if [ -f "/root/.ssh/${SSH_KEY_NAME}" ]; then
    echo "🔑 Found SSH key '${SSH_KEY_NAME}', adding to agent..."
    ssh-add /root/.ssh/${SSH_KEY_NAME}
    echo "✅ Added SSH key to agent"
else
    echo "⚠️ SSH key '/root/.ssh/${SSH_KEY_NAME}' not found"
fi

# Test SSH connection
echo "🔍 Testing SSH connection to GitHub..."
ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
ssh -T git@github.com 2>&1 || true

# Create NVIDIA device symlinks if nvidia-ctk is available
if command -v nvidia-ctk &>/dev/null; then
    echo "🔧 Creating NVIDIA device symlinks..."
    nvidia-ctk system create-dev-char-symlinks --create-all || true
fi

# Simplified workspace approach - always use PROJECT_NAME
if [ ! -z "${PROJECT_NAME}" ]; then
    echo "🚀 Using workspace directory: /workspace/${PROJECT_NAME}"
    cd /workspace/${PROJECT_NAME}
fi

# Print system information
echo "🖥️  Container System Information:"
if [[ "$(uname -s)" == "Linux" ]]; then
    # Report variant information
    echo "🧩 Using variants: ${SYSTEM_VARIANT:-cuda}/${ENV_VARIANT:-default}"

    # Only run CUDA checks on Linux
    if command -v nvidia-smi &>/dev/null; then
        echo "✅ CUDA is available"
        nvidia-smi
        # Get number of available GPUs
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        echo "📊 Found ${GPU_COUNT} GPU(s)"
        # Set CUDA_VISIBLE_DEVICES to all available GPUs (0,1,2,etc.)
        export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT - 1)))
        echo "🎯 CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
    else
        echo "⚠️  WARNING: CUDA is not available"
    fi
else
    echo "🍎 Running on macOS"
fi
echo "-----------------------------------"

# Setup Git configuration
echo "🔧 Setting up git configuration..."
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1
git config --global --replace-all user.email "${GIT_EMAIL}" || true
git config --global --replace-all user.name "${GIT_NAME}" || true
git config --global --replace-all safe.directory '*' || true

# Save SSH agent environment variables to a file
echo "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK" >/etc/profile.d/ssh-agent.sh
echo "export SSH_AGENT_PID=$SSH_AGENT_PID" >>/etc/profile.d/ssh-agent.sh
chmod +x /etc/profile.d/ssh-agent.sh

# Source SSH environment in shell config files
echo "source /etc/profile.d/ssh-agent.sh" >>/root/.bashrc
if [ -f /root/.zshrc ]; then
    echo "source /etc/profile.d/ssh-agent.sh" >>/root/.zshrc
fi

# Register with Ray cluster if RAY_HEAD_ADDRESS is set
if [ ! -z "${RAY_HEAD_ADDRESS}" ]; then
    # Check if the ray-init.sh script exists
    if [ -f /usr/local/bin/ray-init.sh ]; then
        echo "🌟 Registering with Ray cluster..."
        /usr/local/bin/ray-init.sh &
    else
        echo "⚠️ Ray initialization script not found, skipping Ray registration"
    fi
fi

# Execute the command passed to docker run
exec "$@"
