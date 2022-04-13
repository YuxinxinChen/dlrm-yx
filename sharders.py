_sharders = {}

def get_splits(T, ndevices):
    k, m = divmod(T, ndevices)
    if m == 0:
        splits = [k] * ndevices
    else:
        splits = [(k + 1) if i < m else k for i in range(ndevices)]
    return splits


def register_sharder(sharder_name):
    def decorate(func):
        _sharders[sharder_name] = func
        return func
    return decorate


# get device indices for tables
# e.g 8 tables, No. [1,3,5,6] on device 0, No. [2,4,7,8] on device 1, then
# return [0, 1, 0, 1, 0, 0, 1, 1]
# N.B.: only for single-node multi-GPU for now.
def shard(Es, ndevices, alg="naive"):
    if alg not in _sharders:
        import sys
        sys.exit("ERROR: sharder not found")
    return _sharders[alg](Es, ndevices)


@register_sharder("naive")
def naive_shard(Es, ndevices):
    return [(x % ndevices) for x in range(len(Es))]


@register_sharder("naive_chunk")
def naive_shard(Es, ndevices):
    T = len(Es)
    splits = get_splits(T, ndevices)
    table_device_indices = []
    for idx, s in enumerate(splits):
        table_device_indices.extend([idx] * s)
    return table_device_indices


@register_sharder("greedy")
def greedy_shard(Es, ndevices):
    T = len(Es)
    buckets = [0] * ndevices
    table_device_indices = [0] * T # Mapping of embedding tables to devices
    for k, E in enumerate(Es):
        device_idx = buckets.index(min(buckets))
        buckets[device_idx] += E
        table_device_indices[k] = device_idx # Which table goes to which device
    return table_device_indices


@register_sharder("hardcode")
def hardcode_shard(Es, ndevices):
    T = len(Es)
    return [0] + [1] * (T-1)
    #return [0] * (T-1) + [1]
    #return [0] * int(T/2) + [1] * int(T/2)
