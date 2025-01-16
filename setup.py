from setuptools import setup, find_packages

setup(
    name="KSSDS",
    version="1.0.6",
    author="Gun Yang",
    author_email="ggomarobot@gmail.com",
    description="Korean Sentence Splitter for Dialogue Systems",
    long_description=open("README_PYPI.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ggomarobot/KSSDS",
    packages=find_packages(where="src", include=["KSSDS*"]),  # "KSSDS"와 하위 패키지만 포함
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "GPUtil>=1.4,<2.0",
        "numpy>=1.19.5,<2.0",
        "PyYAML>=6.0,<7.0",
        "scikit_learn>=1.6,<2.0",
        "torch>=2.5,<3.0",
        "transformers>=4.42,<5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="MIT",
    keywords="Korean NLP sentence splitter dialogue systems",
)