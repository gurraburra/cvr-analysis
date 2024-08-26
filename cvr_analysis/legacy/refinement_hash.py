import hashlib

def refinementDict(outlier_filter, outlier_threshold, normalization):
    refinement_info = {
            "outlier_filter"                         : outlier_filter,
            "outlier_threshold"                      : outlier_threshold,
            "normalization"                          : normalization,
        }
    # analysis id
    analysis_id = hashlib.sha1(str(tuple(refinement_info.items())).encode("UTF-8")).hexdigest()[:7]
    # save analysis info
    return {**{"refinement_id" : analysis_id}, **refinement_info}