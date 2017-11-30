import sys, os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
#build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

additional_imports = ['numpy.core._methods', 'numpy.lib.format', "matplotlib.backends.backend_tkagg",
                      'scipy.spatial', 'tkinter']
                      #'TKinter', 'six']

packages = ['matplotlib', 'scipy', 'scipy.spatial', 'tkinter']


include_files = [r"C:\Python34\DLLs\tcl86t.dll",
                 r"C:\Python34\DLLs\tk86t.dll"]

os.environ['TCL_LIBRARY'] = r'C:\Python34\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python34\tcl\tk8.6'

setup(name="BinConverterGUI",
      version="1.0",
      description="BinConverterGUI converts the raw .bin files produced by DacqUSB to the normal Tint format." +
      "This will allow the user to select a threshold to use retroactively.",
      options={"build_exe": {'packages': packages,
                             "include_files": include_files,
                             'includes': additional_imports}},
      executables=[Executable("BinConverterGUI.py", base=base)])