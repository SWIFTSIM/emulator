import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("swiftemulator/__version__.py", "r") as fh:
    exec_output = {}
    exec(fh.read(), exec_output)
    __version__ = exec_output["__version__"]

setuptools.setup(
    name="swiftemulator",
    version=__version__,
    description="Gaussian process emulator for creating synthetic model data across high dimensional parameter spaces, initially developed for use with the SWIFT simulation code.",
    url="https://github.com/SWIFTSIM/emulator",
    author="Josh Borrow",
    author_email="josh@joshborrow.com",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "unyt",
        "attrs",
        "george",
        "SALib",
        "scikit-learn",
        "corner",
        "velociraptor",
        "pyyaml",
        "pyDOE",
        "tqdm",
        "emcee",
    ],
    include_package_data=True,
)
