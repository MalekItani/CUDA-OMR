# This is a CUDA Accelerated Optical Music Recognition system, a course project for EECE 696 (Applied Parallel Programming)

This repository contains three main files. The python file was used for prototyping.
Then, the Python code was re-written in C++, and then parts of the code were accelerated using the CUDA API.
The CPU implementation is found in main.cpp and the CUDA implementation in main.cu.

The tests folder contains a class to play output files using the mingus library (work in progress).

## References
[1] Jorge Calvo Zaragoza, Jan Hajic Jr, and Alexander Pacha. Understandingoptical music recognition.arXiv preprint arXiv:1908.03608, 2019.

[2] CATHERINE SCHMIDT-JONES.UNDERSTANDING BASIC MUSICTHEORY.  12TH MEDIA SERVICES, 2018.

[3] Ana Rebelo,  Ichiro Fujinaga,  Filipe Paszkiewicz,  Andr ́e Mar ̧cal,  CarlosGuedes,  and  Jaime  Cardoso.   Optical  music  recognition:   State-of-the-art  and  open  issues.International Journal of Multimedia InformationRetrieval, 1, 10 2012.

[4] Rafael C.. Gonzalez and Richard E.. Woods.Digital image processing.Pearson., 2018.

[5] Pierfrancesco Bellini, Ivan Bruno, and Paolo Nesi.  Optical music sheetsegmentation.  pages 183 – 190, 12 2001.

[6] Florence Rossant and Bloch Isabelle.  Robust and adaptive omr systemincluding fuzzy modeling, fusion of musical rules, and possible error de-tection.EURASIP Journal on Advances in Signal Processing, 2007, 012007.

[7] Ichiro Fujinaga.Optical music recognition using projections. PhD thesis,McGill University Montreal, Canada, 1988.

[8] Khushboo Khurana and Reetu Awasthi. Techniques for object recognitionin images and multi-object detection.International Journal of AdvancedResearch in Computer Engineering & Technology (IJARCET), 2(4):1383–1388, 2013
