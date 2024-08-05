from glob import glob
import re
import os
from datetime import datetime
import pandas as pd

def getOutputFiles(path, include_run = True):
    analysis_list = []
    refinement_info_exist = None
    for file in glob(path):
        row = []
        row.append(re.search("sub-(..)_", file).group(1))
        row.append(re.search("ses-(..)_", file).group(1))
        if include_run:
            row.append(re.search("run-(..)_", file).group(1))
        row.append(re.search("task-([^_]*)_", file).group(1))
        row.append(re.search("_analys-([^_]*)_", file).group(1))
        try:
            row.append(re.search("_refinement-([^_]*)_", file).group(1))
            if refinement_info_exist == False:
                raise RuntimeError("Inconistent refinement")
            refinement_info_exist = True
        except:
            if refinement_info_exist == True:
                raise RuntimeError("Inconistent refinement")
            else:
                refinement_info_exist = False
        row.append(file)
        row.append(datetime.fromtimestamp(os.path.getctime(file)))
        analysis_list.append(row)
    
    # columns
    columns=["subject", "session"]
    if include_run:
        columns.append("run")
    columns += ["task", "analysis"]
    if refinement_info_exist == True:
        columns.append("refinement")
    columns += ["file", "date"]

    return pd.DataFrame(analysis_list, columns=columns)
    