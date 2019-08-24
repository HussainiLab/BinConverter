# BinConverter

This is a GUI developed to aid those individuals using Axona' LLC's Tint software for spike sorting. The GUI will take the raw (.bin) files and convert them to the Tint format (.eeg, .egf, .N, etc) retroactively. This can be done using DacqUSB, however this GUI will allow you to batch convert, as well as it will allow you to retroactively choose a threshold. This is beneficial as depending on the timeframe of your recording, the signal can deviate and the threshold you set in the beginning of the recording might no longer be suitable. As of now the threshold is a multiple (chosen by the user) of the standard deviation of the baseline signal.



### Requirements
- Python: This code was written using Python 3.7, however the compatibility really only depends on PyQt5, which requires Python 2.6+. This was written in Python 3 so I suggest getting the latest version of Python2. It will make the installation process easier as PyQt5 used to be a pain to download in the older versions (3.4 for example). If you happen to have problems downloading PyQ5t, you will need to search for a wheel (.whl) file for PyQt5 for your version (of Python and OS).
- Operating System: BinConverter does not technically have any Operating System (OS) requirements. However you have the ability to sort the newly converted data using Tint (KlustaKwik), and Tint does require Windows3. Some Linux users have been starting to use this, so I have been working out any bugs that maybe be caused by Linux. The .set file is encoded in cp1252, which Python does not default to when reading text files.

### Python Dependencies
- BinConverter
- BatchTINTV3
- NumPy
- PyQt5
- PyQt5-sip
- SciPy

### Project Documentation
- [Installation](https://geba.technology/project/binconverter)
- [User Guide](https://geba.technology/project/binconverter-binconverter-user-guide)


## Authors
* **Geoff Barrett** - [Geoff’s GitHub](https://github.com/GeoffBarrett)

## License
This project is licensed under the GNU  General  Public  License - see the [LICENSE.md](LICENSE.md) file for details
