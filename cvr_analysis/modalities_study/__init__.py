from .workflows.main_wf import cvr_wf
from .workflows.post_processing import data_loader_wf, signal_processing_wf
from .workflows.regression import setup_regression_wf, iterative_regression_wf

data_loader_wf.cacheNodesData(True)
signal_processing_wf.cacheNodesData(True)
setup_regression_wf.cacheNodesData(True)
iterative_regression_wf.cacheNodesData(True)