import setuptools
from reliable_conformal_prediction import __version__

with open("README.md", "r", encoding='ascii') as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding='ascii') as fr:
    required = fr.read().splitlines()

setuptools.setup(
    name="reliable_conformal_prediction",
    version=__version__,
    author="M. Sánchez-Domínguez, J. de Vicente, G. Rubio, E. Valero, L. Lacasa",
    author_email="miguel.sanchez.dominguez@upm.es",
    description="Small package for reliable conformal prediction methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Miguel-San/Reliable-Conformal-Prediction",
    packages=setuptools.find_packages(include = ["reliable_conformal_prediction*"]),
    install_requires=required,
    license_files= ('LICENSE',)
)