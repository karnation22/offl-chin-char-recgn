# Offline Chinese Character Recognition
Please read the details below:

Prerequisite programming language: 
 -- Python 3.7.

Install the following dependencies: 
 - 1 - CASIA 
 - 2 - PIL 
 - 3 - logging
 - 4 - resizeimage
 - 5 - numpy
 - 6 - skimage
 - 7 - argparse
 - 8 - torch and torchvision (pytorch)
 - 9 - collections
 - 10 - statistics
 - 11 - random
 - 12 - matplotlib
 - 13 - svm
 - 14 - sklearn
 - 15 - svmutil
 - 16 - pickle
      
- Iterate in the following order (source or terminal): (python ___)

  - 1. Run data_collection.py; yield "chin_char_trn", "chin_char_cv", "chin_char_tst"      
  
  - 2. Run data_preprocessing.py; yield "chin_char_trn_preproc", "chin_char_cv_preproc", "chin_char_tst_preproc"
  
  - 3. Run data_training_cnn.py (for CNN models):
  
        a) Type '-h' to see terminal arguments. 
        
        b) Other hyperparameters are capitalized near the top of the code. 
        
        c) Yield 6 CNN plots in  the ".png" files, and 6 CNN models in the ".dat" files all in "torch_cnn_data".
        
  - 4. Run data_training_svm.py (for SVM models); yield two SVM models (LIBSVM in .model and Sklearn in .pkl)
  
        -- Arguments are NUM_CLASSES(default=200), NUM_PTS_PER_CLASS_1(default=100), and NUM_PTS_PER_CLASS_2(default=20).
        
       
DISCLAIMER: The following implementation may contain errors or may not work in your particular environment. 

            If so, feel free to post issues on the issue thread.
