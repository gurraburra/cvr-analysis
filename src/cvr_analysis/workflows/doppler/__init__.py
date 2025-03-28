from cvr_analysis.workflows.doppler.cvr_wf import cvr_wf
from cvr_analysis.workflows.doppler.post_processing import data_loader_wf, signal_processing_wf
from cvr_analysis.workflows.doppler.regression import setup_regression_wf, iterate_cvr_wf

data_loader_wf.cacheNodesData(True)
signal_processing_wf.cacheNodesData(True)
setup_regression_wf.cacheNodesData(True)
iterate_cvr_wf.cacheNodesData(True)

__all__ = ["cvr_wf"]