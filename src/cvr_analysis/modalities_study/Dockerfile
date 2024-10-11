FROM python:3.13 
ADD https://api.github.com/repos/gurraburra/cvr-analysis/git/refs/heads/dry-paper cvr-analysis-version.json
RUN pip install git+https://github.com/gurraburra/cvr-analysis.git@dry-paper
ENTRYPOINT ["cvr-modality-study"]