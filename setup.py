from setuptools import setup, find_packages

setup(
    name="hw_gridworld",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.6",
    author="Shashank Bassi",
    author_email="shashankbassi92@gmail.com",
    description="Gridworld assignment",
    tests_require=["pytest"],
    install_requires=[
        "numpy"
    ]
)
