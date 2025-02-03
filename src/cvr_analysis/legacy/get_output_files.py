from glob import glob
import re
import os
from datetime import datetime
import pandas as pd

def getOutputFiles(path):
    analysis_list = []

    optional_fields = {field : None for field in ["ses", "task", "run", "space", "analys", "refinement"]}
    field_full_name = {"ses" : "session", "analys" : "analysis"}

    def checkOptionalField(field, file, row):
        try:
            row.append(re.search(f"_{field}-([^_]*)_", file).group(1))
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
        row.append(re.search("/sub-(..)_", file).group(1))
        for field in optional_fields.keys():
            checkOptionalField(field, file, row)
        row.append(file)
        row.append(datetime.fromtimestamp(os.path.getctime(file)))
        analysis_list.append(row)
    
    # columns
    columns=["subject"]
    for k,v in optional_fields.items():
        if v:
            if k in field_full_name:
                columns.append(field_full_name[k])
            else:
                columns.append(k)
    columns.extend(["file", "date"])

    return pd.DataFrame(analysis_list, columns=columns)
    