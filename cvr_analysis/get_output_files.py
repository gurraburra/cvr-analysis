from glob import glob
import re
import os
from datetime import datetime
import pandas as pd

def getOutputFiles(path):
    analysis_list = []

    optional_fields = {field : None for field in ["refinement", "task", "run"]}

    def checkOptionalField(field, regex, file, row):
        try:
            row.append(re.search(regex, file).group(1))
            if optional_fields[field] == False:
                raise RuntimeError(f"Inconistent field: {field}")
            optional_fields[field] = True
        except:
            if optional_fields[field] == True:
                raise RuntimeError(f"Inconistent field: {field}")
            else:
                optional_fields[field] = False

    for file in glob(path):
        row = []
        row.append(re.search("sub-(..)_", file).group(1))
        row.append(re.search("ses-(..)_", file).group(1))
        checkOptionalField("task", "task-([^_]*)_", file, row)
        checkOptionalField("run", "run-(..)_", file, row)
        row.append(re.search("analys-([^_]*)_", file).group(1))
        checkOptionalField("refinement", "refinement-([^_]*)_", file, row)
        row.append(file)
        row.append(datetime.fromtimestamp(os.path.getctime(file)))
        analysis_list.append(row)
    
    # columns
    columns=["subject", "session"]
    if optional_fields["task"]:
        columns.append("task")
    if optional_fields["run"]:
        columns.append("run")
    columns.append("analysis")
    if optional_fields["refinement"]:
        columns.append("refinement")
    columns.extend(["file", "date"])

    return pd.DataFrame(analysis_list, columns=columns)
    