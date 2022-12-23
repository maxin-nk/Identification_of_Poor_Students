# identification of poor students

# train
The folder includes datasets for poor student identification. It has six original datasets, BrData, CdData, DmData, LbData, AcData, and SbData, which are the data left by 10885 students (1560 poor students) in the Academic Affairs Management System, Campus Card System, and Financial Aid Management System during 2019/09–2021/09. Each dataset's data has been mixed together over time, and some of it is duplicate or abnormally dirty data. Among them, BrData stores 239,947 student borrowing records. 124,555,558 student spending records are stored in CdData. 211,564 student dormitory access card records are stored in DmData. 101,27447 student library access card records are stored in LbData. 9,130 student grade records are stored in AcData. SbData stores the student's secondary amounts in four categories: 0 RMB (ordinary student), 1000 RMB (generally poor student), 1500 RMB (mediumly poor student), and 2000 RMB (extremely poor student), which are data labels. In particular, in order to ensure the accuracy and validity of the evaluation, we verified the financial aid amounts in SbData one by one in nearly six months by using online research as the main method, supplemented by field visits (affected by the new crown epidemic), and those that did not match the real poverty level of students were adjusted.

# Apriori+KNN.py
Apriori algorithm for identifying poor students.

# BP neural network.py
BP algorithm for identifying poor students.

# GaussianMixture.py
GaussianMixture clustering algorithm for identifying poor students.

# K-means.py
K-means clustering algorithm for identifying poor students.

# Random Forest.py
Random Forest algorithm for identifying poor students.

# XGBoost.py
XGBoost algorithm for identifying poor students.
