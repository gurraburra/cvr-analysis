FROM python:3.13 
WORKDIR /app
ADD https://api.github.com/repos/gurraburra/cvr-analysis/git/refs/heads/main cvr-analysis-version.json
RUN pip install git+https://github.com/gurraburra/cvr-analysis.git@main
ENTRYPOINT ["cvr-analysis"]