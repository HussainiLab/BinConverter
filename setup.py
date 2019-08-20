import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages()
print('found these packages:', pkgs)

pkg_name = "BinConverter"

setuptools.setup(
    name=pkg_name,
    version="1.0.2",
    author="Geoffrey Barrett",
    author_email="geoffrey.m.barrett@gmail.com",
    description="BinConverter - GUI that will allow the user to convert the raw (.bin) files from Axona's " +
                "Tint software to their normal Tint format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HussainiLab/BinConverter.git",
    packages=pkgs,
    install_requires=
    [
        'PyQt5',
        'BatchTINTV3',
        'numpy',
        'scipy'
    ],
    package_data={'BinConverter': ['img/*.png']},
    classifiers=[
        "Programming Language :: Python :: 3.7 ",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
    ],
)
