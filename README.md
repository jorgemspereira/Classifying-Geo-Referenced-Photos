# Classifying Geo-Referenced Photos and Satellite Images for Supporting Terrain Classification  

The source code presented in this repository leverages the keras.io deep learning library (combined with scikit-learn, and other machine learning libraries) to test the usability of the DenseNet neural architecture (Huang et al. 2017) to classify images regarding the presence of a flood, and the severity of that same flood.

    @inproceedings{huang2017densenet, 
        author    = {G. Huang and Z. Liu and L. v. d. Maaten and K. Q. Weinberger}, 
        booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition}, 
        title     = {Densely Connected Convolutional Networks}, 
        year      = {2017}
    }


(More details later...)

### Files needed not in this repository

The dataset files are too big to put on a github repository, so it is necessary to download them and place them in the following hierarchy:

```
project
│   README.md
│   main.py
│   ...
└─── datasets
│   └─── EuropeanFlood2013
│   │   │    classification.csv
│   │   │    metadata.json
│   │   └─── imgs_small
│   │       │    25441112.jpg 
│   │       │    25441113.jpg 
│   │       │    ...
│   │   
│   └─── FloodSeverity 
│   │   │    dataset_european_flood_2013.csv
│   │   │    dataset_test_mediaeval2017.csv
│   │   │    dataset_test_mediaeval2018.csv
│   │   │    dataset_train_mediaeval2017.csv
│   │   │    dataset_train_mediaeval2018.csv
│   │
│   └─── MediaEval2017
│   │   └─── Classification 
│   │       └─── development_set
│   │       │   │    devset_images_gt.csv
│   │       │   │    devset_images_metadata.json
│   │       │   └─── devset_images
│   │       │       │    224249.jpg
│   │       │       │    228743.jpg
│   │       │       │    ...
│   │       │
│   │       └─── test_set
│   │           │    testset_images_gt.csv
│   │           │    testset_images_metadata.json
│   │           └─── testset_images
│   │               │    182740.jpg
│   │               │    363897.jpg
│   │               │    ...
│   │       
│   └─── MediaEval2018  
│   │   └─── Classification 
│   │       └─── development_set
│   │       │   │    devset_images_gt.csv
│   │       │   └─── devset_images
│   │       │       │    900095331450273792.jpg
│   │       │       │    900160936945831936.jpg
│   │       │       │    ...
│   │       │
│   │       └─── test_set
│   │           │    testset_images_gt.csv
│   │           └─── testset_images
│   │               │    897989160002256901.jpg
│   │               │    898024131051921408.jpg
│   │               │    ...
```

The "European Floods 2013" dataset can be obtained from:   
https://github.com/cvjena/eu-flood-dataset  

The dataset from "Multimedia Satelite Task from MediaEval 2017" can be obtained from:  
https://github.com/multimediaeval/2017-Multimedia-Satellite-Task/wiki/Data-Release  

The dataset from "Multimedia Satelite Task from MediaEval 2018" can be obtained from:  
https://github.com/jorgemspereira/MediaEval2018-Image-Downloader  

The Flood Severity labels can be found here:  
https://github.com/jorgemspereira/Flood-Image-Tagger/tree/master/results  

### How to use  

The code was developed and tested in Python 3.6.7 with Keras 2.2.4, using Tensorflow as backend. The code supports re-training in cross-validation, and train-test split (being that the test split will be the test split from the MediaEval 2017 DIRSM task). Also, it is possible to re-evaluate a previous saved model. To run the script simple execute:

```console
$ python3 main.py --mode {train, load} --model {dense_net, attention_guided} --dataset {mediaeval2017, mediaeval2018, european_floods, both, flood_severity_3_classes, flood_severity_european_floods} --method {cross_validation, train_test_split} --data-augmentation --class-activation-map --print-classifications
```

Where the flag _data-augmentation_ will use data-augmentation to train the models; the flag _class-activation-map_ will draw the class activation maps for each test example; and _print-classifications_ will print the classification for each test example and save them in a file on the root directory.
