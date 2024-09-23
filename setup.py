import setuptools

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="value_aggregation",
    version="0.0.1",
    author="Jared Moore",
    author_email="jaredm@allenai.org",
    description="Value Aggregation: Models of How to Compromise",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
)
