
_sharders = {}

def register_sharder(sharder_name):
    def decorate(func):
        _sharders[sharder_name] = func
        return func
    return decorate

# get device indices for tables
# e.g 8 tables, No. [1,3,5,6] on device 0, No. [2,4,7,8] on device 1, then
# return [0, 1, 0, 1, 0, 0, 1, 1]
def shard(T, Es, ndevices, alg="naive"):
    if alg not in _sharders:
        import sys
        sys.exit("ERROR: sharder not found")
    return _sharders[alg](T, Es, ndevices)

@register_sharder("naive")
def naive_shard(T, Es, ndevices):
    return [(x % ndevices) for x in Es]

@register_sharder("greedy")
def greedy_shard(T, Es, ndevices):
    buckets = [0] * ndevices
    table_device_indices = [0] * T # Mapping of embedding tables to devices
    for k, E in enumerate(Es):
        device_idx = buckets.index(min(buckets))
        buckets[device_idx] += E
        table_device_indices[k] = device_idx # Which table goes to which device
    return table_device_indices

@register_sharder("hardcode")
def hardcode_shard(T, Es, ndevices):
    #return [0] + [1] * (T-1)
    #return [0] * (T-1) + [1]
    return [0] * int(T/2) + [1] * int(T/2)

