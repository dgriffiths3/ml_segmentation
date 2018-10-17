# Machine Learning - Image Segmentation

Per pixel image segmentation using machine learning algorithms. Programmed using the following libraries: Scikit-Learn, Scikit-Image OpenCV, and Mahotas and ProgressBar. Compatible with Python 2.7+ and 3.X.

### Feature vector

Spectral:

* Red
* Green
* Blue

Texture:

* Local binary pattern

Haralick (Co-occurance matrix) features (Also texture):

* Angular second moment
* Contrast
* Correlation
* Sum of Square: variance
* Inverse difference moment
* Sum average
* Sum variance
* Sum entropy
* Entropy

### Supported Learners

* Support Vector Machine
* Random Forest
* Gradient Boosting Classifier


### Example Usage

python train.py -i <path_to_image_folder> -l <path_to_label_folder> -c <SVM, RF, GBC> -o <path/to/model.p>

### Example Output

[[https://github.com/dgriffiths3/ml_segmentation/tree/master/pots/image.png|alt=example]]

![alt text][image]

[image]: https://github.com/dgriffiths3/ml_segmentation/tree/master/pots/image.png "Example Output"
