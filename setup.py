from setuptools import setup, find_packages
from os import path

ROOT_DIR = path.abspath(path.dirname(__file__))

with open(path.join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fxutil",
    version="0.1alpha",
    description=("handy utils"),
    long_description=long_description,
    # url='',
    keywords="utilities, python, science",
    # classifiers=[
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3 :: Only',
    #     'Development Status :: 4 - Beta'
    # ],
    packages=find_packages(),
    # entry_points={"console_scripts": ["paperfridge = paperfridge:start_paperfridge"]},
    install_requires=["matplotlib", "black"],
    author="Felix Jung",
    author_email="jung@posteo.de",
)
