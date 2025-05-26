ARG PYTHON_VERSION
ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME
ARG VARIANT=cuda
ARG ENV_VARIANT=default

FROM ghcr.io/${GIT_NAME}/ml-base:py${PYTHON_VERSION}-${VARIANT}

# Redefine ARGs to make them available after FROM
ARG PYTHON_VERSION
ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME
ARG VARIANT
ARG ENV_VARIANT
ARG SSH_KEY_NAME

# Define ENV vars to make ARG values available at runtime
ENV PATH="/usr/local/bin:/root/.local/bin:/opt/conda/bin:${PATH}"

WORKDIR /workspace/${PROJECT_NAME}

# Setup Git configuration
RUN git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}"

# COPY all requirements files
COPY requirements*.txt ./

RUN uv pip install --system --no-cache-dir "ray[default]==2.44.0"

# Install system variant specific requirements if available
RUN if [ -f "requirements-${VARIANT}.txt" ]; then \
    echo "Installing ${VARIANT} specific requirements..." && \
    uv pip install --system --no-cache-dir -r "requirements-${VARIANT}.txt"; \
    else \
    echo "No ${VARIANT} specific requirements file found"; \
    fi

# Install environment variant specific requirements if available and not default
RUN if [ "${ENV_VARIANT}" != "default" ] && [ -f "requirements-${ENV_VARIANT}.txt" ]; then \
    echo "Installing ${ENV_VARIANT} specific requirements..." && \
    uv pip install --system --no-cache-dir -r "requirements-${ENV_VARIANT}.txt"; \
    elif [ "${ENV_VARIANT}" != "default" ]; then \
    echo "No ${ENV_VARIANT} specific requirements file found"; \
    fi

# Always install ipdb for debugging (system-wide)
RUN uv pip install --system --no-cache-dir ipdb

COPY scripts/ray-init.sh /usr/local/bin/ray-init.sh
RUN chmod +x /usr/local/bin/ray-init.sh

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["zsh"]