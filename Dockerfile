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

# Copy project files for uv
COPY pyproject.toml uv.lock ./

# Install dependencies using uv with appropriate extras
RUN echo "Installing dependencies with uv..." && \
    if [ "${VARIANT}" = "cpu" ]; then \
        uv sync --locked --extra dev --extra cpu; \
    elif [ "${VARIANT}" = "cuda" ]; then \
        uv sync --locked --extra dev --extra cuda; \
    elif [ "${VARIANT}" = "cuda-nightly" ]; then \
        uv sync --locked --extra dev --extra cuda-nightly; \
    else \
        uv sync --locked --extra dev --extra cuda; \
    fi

# Install ray if needed for the environment variant
RUN if [ "${ENV_VARIANT}" = "ray" ]; then \
        echo "Installing ray extras..." && \
        uv sync --locked --extra ray; \
    fi

# Always install ipdb for debugging
RUN uv pip install --system ipdb

COPY scripts/ray-init.sh /usr/local/bin/ray-init.sh
RUN chmod +x /usr/local/bin/ray-init.sh

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["zsh"]