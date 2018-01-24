## Content of project:
- _GenderLabellingFinal.csv_: the final result file in csv format, first column is customer_id, laster column is gender(female=0/male=1)
- _genderclassifier.ipynb_ : the main jupyter note file
- _genderclassifier.py_: the main python file
- _genderclassifierfunctions.py_: the python file which provides all the functions
- _test_data.zip_: the data file provided
- _data.json_: the json file extracted from test_data.zip

## How to run
### Method 1: Jupyter Notebook
- type following command from consol:

 `git clone https://github.com/CarlFYZ/UnsupervisedGenderLabelling.git`
 
 `cd UnsupervisedGenderLabelling/`
 
 `jupyter notebook`
 
 `click genderclassifier.ipynb in browser`


### Method 2: Execute the program directly
-  (not recommended, jupyter notebook provides more infomation)

  `git clone https://github.com/CarlFYZ/UnsupervisedGenderLabelling.git`
  
  `cd UnsupervisedGenderLabelling/`
  
  `python.exe genderclassifier.py`

### Method 3: From PyCharm
- (not recommended, jupyter notebook provides more infomation)

  `git clone https://github.com/CarlFYZ/UnsupervisedGenderLabelling.git`
  
  `Open project UnsupervisedGenderLabelling from PyCharm`
  
  `select genderclassifier.py and click run from menu (may need to set path of python)`
  
## Dependencies
- python 3+
- PyCharm 2017(optional, only if your run from PyCharm)
- python libraries: see import statement in _genderclassifierfunctions.py_ and _genderclassifier.py_
    
