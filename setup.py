from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name='Notq',
    version='1.0.0',
    description='Notq is a Python base tool collected and developed for speech and language processing in Persian',
    long_description=readme,
    author='Nbic',
    long_description_content_type="text/markdown",
    packages=find_packages(include=["Notq*"]),
    url="https://github.com/shaqayeql/Notq",
    install_requires=['torchaudio ==0.9.0',
                        'pydub ==0.25.1',
                        'speechRecognition ==3.8.1',
                        'numpy ==1.19.2',
                        'tqdm ==4.61.2',
                        'transformers ==4.11.2',
                        'torch ==1.9.0',
                        'wget ==3.2'],

    keywords=['python', 'first package'],
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
        ]
)