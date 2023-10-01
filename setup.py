from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


reqs = read_requirements("requirements.txt")




setup(
    name="docstring_writer",
    version="0.0.1",
    long_description_content_type="text/markdown",
    packages=find_packages(
        exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']
    ),
    install_requires=reqs,
    python_requires=">=3.8",
)
