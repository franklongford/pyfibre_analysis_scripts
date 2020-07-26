import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()


setuptools.setup(
    name="pyfibre_analysis_tools",
    version="0.0.1",
    author="Frank Longford",
    author_email="franklongford@gmail.com",
    description="Analysis scipts for PyFibre databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/franklongford/pyfibre_analysis_scripts",
    packages=setuptools.find_packages(),
    requirements=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
