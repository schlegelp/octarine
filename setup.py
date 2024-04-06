import re

from setuptools import setup, find_packages
from extreqs import parse_requirement_files
from pathlib import Path

VERSIONFILE = "octarine/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

HERE = Path(__file__).resolve().parent
install_requires, extras_require = parse_requirement_files(
    HERE / "requirements.txt",
)


setup(
    name='octarine3d',
    version=verstr,
    packages=find_packages(),
    license='BSD-2-Clause',
    description='WGPU-based 3d viewer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/schlegelp/octarine',
    project_urls={
     "Documentation": "https://schlegelp.github.io/octarine/",
     "Source": "https://github.com/schlegelp/octarine",
     "Changelog": "https://github.com/schlegelp/octarine/releases",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='3D viewer WGPU pygfx',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=install_requires,
    extras_require=dict(extras_require),
    python_requires='>=3.8',
    zip_safe=False,
    include_package_data=True
)
