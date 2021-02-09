import setuptools
from swiftemulator import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swiftemulator",
    version=__version__,
    description="Emulator for interpolating SWIFT runs.",
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
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "unyt", "attrs", "george", "SALib", "scikit-learn"],
    include_package_data=True,
)
