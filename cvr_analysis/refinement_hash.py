import hashlib

def refinementDict(filter_type, filter_value, normalization):
    refinement_info = {
            "filter_type"                       : filter_type,
            "filter_value"                      : filter_value,
            "normalization"                     : normalization,
        }
    # analysis id
    analysis_id = hashlib.sha1(str(tuple(refinement_info.items())).encode("UTF-8")).hexdigest()[:7]
    # save analysis info
    return {**{"refinement_id" : analysis_id}, **refinement_info}