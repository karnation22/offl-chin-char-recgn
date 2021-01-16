# Offline Chinese Character Recognition
Please read the details below:

Prerequisite programming language: 
 -- Python 3.7.

Install the following dependencies: 

 -- 1: CASIA 

 -- 2: PIL 

 -- 3: logging

 -- 4: resizeimage

 -- 5: numpy

 -- 6: skimage

 -- 7: argparse

 -- 8: torch

 -- 9: torchvision (pytorch)

 -- 10: collections

 -- 11: statistics

 -- 12: random

 -- 13: matplotlib

 -- 14: svm

 -- 15: sklearn

 -- 16: svmutil

 -- 17: pickle

 -- 18: gensim

 -- 19: graphlab

 -- 20: nltk
      
- Iterate in the following order (source or terminal): (python ___)

  --- 1. Run data_collection.py; yield "chin_char_trn", "chin_char_cv", "chin_char_tst"      
  
  --- 2. Run data_preprocessing.py; yield "chin_char_trn_preproc", "chin_char_cv_preproc", "chin_char_tst_preproc"
  
  --- 3. Run data_training_cnn.py (for CNN ); yield 6 CNN plots in ".png" files and 6 CNN models in ".dat" files.
        
        ---- Type '--help' to see terminal arguments. 
        
  --- 4. Run data_training_svm.py (for SVM models); yield two SVM models (LIBSVM in .model and Sklearn in .pkl)
        
        ---- Arguments are NUM_CLASSES(default=200), 
             NUM_PTS_PER_CLASS_1(default=100), and NUM_PTS_PER_CLASS_2(default=20).
        
       
DISCLAIMER: The following implementation may contain errors or may not work in your particular environment; 
            if so, feel free to post issue(s) on the issue thread.
