## Acknowledgement
Some of the python code (comment lines or functions) in this project is not used which are some attempts or hints I have done. I must admit that the code refers to the following two items in many places:

1. https://github.com/jadore801120/attention-is-all-you-need-pytorch
2. https://github.com/eladhoffer/seq2seq.pytorch

# MATLAB code
version is R2018.a
1. Need WFDB Toolbox for MATLAB from https://physionet.org/physiotools/matlab/wfdb-app-matlab/
2. run `get_anno` to get annotation
3. run `denoising` to get denoised signal
4. run `segmentation` to segment the denoised signal
5. run `features` to get the final results, format: label, preRR interval, postRR interval, [heartbeat signal]
6. Here I provide a version of the results, you can also try to use different noise reduction methods or segmentation methods.

# Python code
1. All parameters are defined in `config.py`
2. Some package versions： 
    - python: 3.6.8
    - numpy: 1.16.0
    - Pytorch: '1.0.1.post2'
    - cuda: '10.0.130'
    - tqdm: 4.31.1
3. You may need to change the code in `main.py` to set **train_file** and **valid_file** path according to your preference.
4. Since I have modified my code, I'm not sure the code will work correctly.  So if there is any bug, please let me know.

# License
For academtic and non-commercial usage