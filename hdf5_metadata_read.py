# Class to gather the metadata information stored in a yaml file
# and parse them into a class instance with nested attributes.
# The file generates a global variable with the instance of the
# AttrDict preloaded with the BAHAMAS metadata.
# ======================================================================================
# EXAMPLE IMPLEMENTATION
#
# from hdf5_metadata_read import Metadata
#
# print(Metadata)
# >>> {'data': {'PATHS': {'computername': 'mizar.jb.man.ac.uk', 'dir_hydro': ...
#
# print(Metadata.data)
# >>> {'PATHS': {'computername': 'mizar.jb.man.ac.uk', 'dir_hydro': '/scratch/n ...
#
# print(Metadata.data.HDF5_FILE_STRUCTURE_HYDRO)
# >>> {'subfind_groups_st': {'/': 'group', '/Constants': 'group', '/FOF': 'group', '/F ...
#
# print(Metadata.data.HDF5_FILE_STRUCTURE_HYDRO.subfind_groups_st)
# >>> {'/': 'group', '/Constants': 'group', '/FOF': 'group', '/FOF/ContaminationCount' ...
#
# print(list(Metadata.data.HDF5_FILE_STRUCTURE_HYDRO.subfind_groups_st.keys()))
# >>> ['/', '/Constants', '/FOF', '/FOF/ContaminationCount', '/FOF/Contaminati ...
#
# print([hdfpath \
#        for hdfpath, specifier in Metadata.data.HDF5_FILE_STRUCTURE_HYDRO.subfind_groups_st.items() \
#        if specifier == "dataset"])
# >>> ['/FOF/ContaminationCount', '/FOF/ContaminationMass', '/FOF/FirstSubhaloID ...

import yaml
import os
_dir_config = os.path.dirname(os.path.realpath(__file__))

config_file = os.path.join(_dir_config, 'hdf5_metadata_config.yml')


def walk_dict(d, depth=0):
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        if isinstance(v, dict):
            print("  " * depth + ("%s" % k))
            walk_dict(v, depth + 1)
        else:
            print("  " * depth + "%s %s" % (k, v))


def loadyml():
    metadata = dict()
    with open(config_file) as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)
        for doc in docs:
            metadata = {**metadata, **doc}
    return metadata


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]

    def __setattr__(self, name, value):
        self[name] = self.from_nested_dict(value)

    def __delattr__(self, name):
        if name in self:
            del self[name]

    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key]) for key in data})


def get_metadata() -> AttrDict:
    metadata = AttrDict()
    metadata.data = loadyml()
    return metadata


Metadata = get_metadata()