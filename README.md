# Santander
# Author: I-Ta Lee, lee2226@purdue.edu
# April 5th 2016
#

Quickstart
1. create a directory 'data'
2. put train.csv and test.csv under such directory
3. run training program with the following command

python santander_train.py -c 0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,100,300 data


You can also use this with options below:

$ python santander_train.py -h
usage: santander_train.py [-h] [-k NFOLD] [-o OUTPUT_RESULT] [-m OUTPUT_MODEL]
                          [-c C] [-v] [-d]
                          INPUT_FOLDER

SVM training for Santander

positional arguments:
  INPUT_FOLDER          data folder

optional arguments:
  -h, --help            show this help message and exit
  -k NFOLD, --kfold NFOLD
                        k for kfold cross-validtion. If the value less than 2,
                        we skip the cross-validation and choose the first
                        parameter of -c (DEFAULT: 5)
  -o OUTPUT_RESULT, --output_result OUTPUT_RESULT
                        output, results file name (DEFAULT: result.csv)
  -m OUTPUT_MODEL, --output_model OUTPUT_MODEL
                        output, model file name (DEFAULT: model.pkl)
  -c C, --Cs C          SVM parameter (DEFAULT: 1). This can be a list
                        expression, e.g., 0.1,1,10,100
  -v, --verbose         show messages
  -d, --debug           show debug messages
