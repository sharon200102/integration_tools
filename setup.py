import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.3'
PACKAGE_NAME = 'integration_tools'
AUTHOR = 'Sharon Komissarov'
AUTHOR_EMAIL = 'sharon200102@gmail.com'
URL = 'https://github.com/sharon200102/integration_tools'

LICENSE = ' MIT License'
DESCRIPTION = 'A package in development stages which allows to intgrate few data-sets using machine learning tools'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'torch',
      'pandas',
      'typing',
      'argparse'
      
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
