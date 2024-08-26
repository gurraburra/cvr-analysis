# Define base image
FROM condaforge/mambaforge

# Set working directory for the project
WORKDIR /app

RUN mamba update mamba

# Create Conda environment from the YAML file
COPY environment.yml .
RUN mamba env create -f environment.yml

# Override default shell and use bash
SHELL ["mamba", "run", "-n", "myenv", "/bin/bash", "-c"] 

# get custom packages package
RUN mkdir /custom-packages
# install process control
WORKDIR /custom-packages/
ADD https://api.github.com/repos/gurraburra/process-control/git/refs/heads/main process-control-version.json
RUN git clone --depth 1 https://github.com/gurraburra/process-control.git
WORKDIR /custom-packages/process-control
RUN pip install .

# install cvr analysis
WORKDIR /custom-packages/
ADD https://api.github.com/repos/gurraburra/cvr-analysis/git/refs/heads/main cvr-analysis-version.json
RUN git clone --depth 1 https://github.com/gurraburra/cvr-analysis.git
WORKDIR /custom-packages/cvr-analysis
RUN pip install .

# set workdir back to app
WORKDIR /app

# Python program to run in the container
COPY run.py .
ADD workflows ./workflows

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "myenv", "python", "/app/run.py"]