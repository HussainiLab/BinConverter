# BinConverter

This is a GUI developed to aid those individuals using Axona' LLC's Tint software for spike sorting. The GUI will take the raw (.bin) files and convert them to the Tint format (.eeg, .egf, .N, etc) retroactively. This can be done using DacqUSB, however this GUI will allow you to batch convert, as well as it will allow you to retroactively choose a threshold. This is beneficial as depending on the timeframe of your recording, the signal can deviate and the threshold you set in the beginning of the recording might no longer be suitable. As of now the threshold is a multiple (chosen by the user) of the standard deviation of the baseline signal.

### Prerequisites
The dependencies are written in the [requirements.txt](requirements.txt) file, however some are listed below: 
1) PyQt4 - This is the framework of the GUI so it and its dependents must be installed, I believe a requirement for this is Python 3.4+
2) Numpy
3) SciPy
4) PeakUtils
5) Matplotlib

*Note: it has been tested on Windows, and should be compatible on other OS', however they have not been tested*

### Installing
After you have finished installing the 
1) Clone the BinConverter repository by executing the following command in whichever directory you want BinConverter to be copied to
```
git clone https://github.com/ephyslab/BinConverter.git
```
2) Use desired method to execute BinConverterGUI.py, if you want to execute it from the command prompt use the following command:
```
python "path/to/BinConverterGUI.py"
```
## Authors
* **Geoff Barrett** - [Geoff's GitHub](https://github.com/GeoffBarrett)

## License

This project is licensed under the GNU  General  Public  License - see the [LICENSE.md](LICENSE.md) file for details
