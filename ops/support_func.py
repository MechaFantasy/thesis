
def merge_dict(dict_A, dict_B):
    for k, v in dict_B.items():
        dict_A[k] = v
    return dict_A