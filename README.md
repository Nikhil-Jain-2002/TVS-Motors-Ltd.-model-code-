
### Face Localization using MTCNN model and Face Feature Extraction using Facenet

Steps to be followed:
1) Face Feature Extraction Model to be downloaded from [here](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-). The downloaded model to be unzipped under model directory.
2) Required python 3.7 (more specific v3.7.3).  
3) Required packages can be installed using 'pip install -r requirements.txt' (may be, install in a virtual environment)  
4) Run '$python face_localize_feature_extract.py' on console. 
 
 
Description:

1) 'requirements.txt' -> contains list of packages needed to be installed.
2) 'face_localize_feature_extract.py' -> execute the file to perform face localization and feature extraction.
3) facenet/src/facenet.py -> Required to load (facenet) model to perform facial feature extraction.  
4) facenet/facenet_bk.py -> The original file -- (incase) replace it if the existing facenet.py creates any issues. Also, It can be used for understanding image batch processing.   
5) facenet/align -> code and model to perform MTCNN face localization.
6) LICENECE.md -> MIT Licence --Required to be present.
7) Models/* -> Contains model path to perform facial feature extraction. 


This work is influenced by [Facenet](https://github.com/davidsandberg/facenet)

