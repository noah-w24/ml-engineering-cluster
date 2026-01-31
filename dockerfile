FROM jupyter/base-notebook:lab-4.0.5

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER jovyan
RUN pip install --no-cache-dir jupyterlab-git torch

CMD ["start-notebook.sh"]