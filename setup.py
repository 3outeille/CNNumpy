from glob import glob
from os.path import abspath, basename, dirname, join, normpath, relpath
from shutil import rmtree
from setuptools import setup, find_packages
from setuptools import Command

here = normpath(abspath(dirname(__file__)))
class CleanCommand(Command):
    """
        Custom clean command to tidy up the root folder.
    """
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info ./__pycache__'.split(' ')

    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        global here

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob(normpath(join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print('removing %s' % relpath(path))
                rmtree(path)

setup(  
        cmdclass={'clean': CleanCommand},
        name='CNNumpy',
        version='1.0',
        packages=find_packages()
    )
