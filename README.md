# MLlearningnote
note for starting studying machine learning from 04/18


this is a learning note about tensorflow learning.

## Update at 19.4.09
Update the K-nearest neighbors algorithm on iris dataset, finally achieve 98.3% accuracy.

## update at 18.8.23 
this  repository include 2 achieved part, one is a normal Cnn, which in [normal cnn](https://github.com/JayFu/MLlearningnote/tree/master/normal%20%20Cnn), and another is a faster RCNN, which is edit by dBeker [faster rcnn on windows](https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3.5) and in [faster rcnn on windows](https://github.com/JayFu/MLlearningnote/tree/master/faster%20Rcnn%20on%20Windows)

as for how to implemente this two part, i would introduce bellow.

### normal cnn

this part used to recognize text from captcha, there is a preprocess.py used to preprocess these captcha imgs.

you can find a data_dir at CnnN.py line 55, which used to store training data and somewhere else is validation data, store them in two file folder, and run CnnN.py for training.


after training, predict.py helps on predict the text from captcha img. also, there is data_dir, store imgs in that dir and run some py file call function predict().

### fater rcnn

this part uesd to objection locating. the way to implement this project written by dBeker, you can read the README in the project.

this part can only run on windows, advice environment is tensorflow 1.4 and python 3.5, CUDA8.
