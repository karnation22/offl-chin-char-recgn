# offl_chin_char_recgn
My implementation of Offline Chinese Character Recognition using CASIA Dataset - please read details below.

Prerequisites: 
--Python 3.7 or greater

- Install the following dependencies: 
      1) CASIA
      2) PIL
      3) logging
      4) resizeimage
      5) numpy
      6) skimage
      7) argparse
      8) torch and torchvision (pytorch)
      9) collections
      10) statistics
      11) random
      12) matplotlib
      13) svm
      14) sklearn
      15) svmutil
      16) pickle
      
- Iterate in the following order (source or terminal): (python ___)
  1)  - Run data_collection.py:
      - Yield output "chin_char_trn", "chin_char_cv", "chin_char_tst"
  2)  - Run data_preprocessing.py:
      - Yield "chin_char_trn_preproc", "chin_char_cv_preproc", "chin_char_tst_preproc"
  3a)  - Run data_training_cnn.py (for CNN models). On terminal, arguments include 'batch_size' (default=200), 
         'epochs'(default=20), 'l_rate'(default=0.01), 'l_interval'(default=5), 
         'cv_flag'(default=False). Other hyperparameters are capitalized above. 
       - Yield 6 CNN plots in *.png files and 6 CNN models in *.dat files.
  3b)  - Run data_training_svm.py (for SVM models). On terminal, arguments are NUM_CLASSES(default=200), NUM_PTS_PER_CLASS_1(default=100),
         and NUM_PTS_PER_CLASS_2(default=20).
       - Yield two SVM models (LIBSVM in .model and Sklearn in .pkl)
       
DISCLAIMER: The following implementation is may contain errors or not work in your particular. If so, feel free to post on the issue thread.
