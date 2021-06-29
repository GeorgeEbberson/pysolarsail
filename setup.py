"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="pysolarsail",
    version='0.0.1',
    description="Python package for solar sail simulation and modelling.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/GEbb4/pysolarsail",
    author="George Ebberson",
    author_email="george.ebberson@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords="space, solar sail, simulation",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=[
        "numba>=0.53",
    ],
    extras_require={
        'dev': ['check-manifest'],
        'test': ["coverage", "pytest", "pytest-cov"],
    },
    project_urls={  # Optional
        'Bug Reports': "https://github.com/GEbb4/pysolarsail/issues",
        'Source': "https://github.com/GEbb4/pysolarsail",
    },
)


# package_data = {  # Optional
#                    'sample': ['package_data.dat'],
#                },
#
# # Although 'package_data' is the preferred approach, in some case you may
# # need to place data files outside of your packages. See:
# # http://docs.python.org/distutils/setupscript.html#installing-additional-files
# #
# # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
# data_files = [('my_data', ['data/data_file'])],  # Optional

# # To provide executable scripts, use entry points in preference to the
# # "scripts" keyword. Entry points provide cross-platform support and allow
# # `pip` to create the appropriate form of executable for the target
# # platform.
# #
# # For example, the following would provide a command called `sample` which
# # executes the function `main` from this package when invoked:
# entry_points = {  # Optional
#                    'console_scripts': [
#                        'sample=sample:main',
#                    ],
#                },