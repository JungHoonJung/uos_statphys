from setuptools import setup, find_packages
from uos_statphys import __version__


setup(name='uos_statphys',
		version = __version__,
		description='Python library for statistical physics lab of University of Seoul.',
		author='Junghoon Jung',
		author_email='jh.jung@uos.ac.kr',
		packages=find_packages(),      
		include_package_data=True,      # include files in MANIFEST.in
		python_requires = '>=3.7',
		install_requires=['numpy', 'matplotlib','tqdm', 'clang']
	 )
