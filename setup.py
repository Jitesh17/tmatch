import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tmatch",  
    version="0.0.1",
    author="Jitesh Gosar",
    author_email="gosar95@gmail.com",
    description="Service for demonstration of Template Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jitesh17/tmatch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)