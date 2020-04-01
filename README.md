# Classification
Matlab
### classification.m
Két osztáyt különböztet meg: airplanes és motorbikes  
A Predict függvény értéke alapján az OutputToClass 0 vagy 1-es osztályba sorolja a képeket  
0 -> airplane  
1 -> motorbike
### classification_3classes.m
Három osztályt kölünböztet meg: airplanes, motorbikes és faces  
Az OutputToClass kimenetei:  
[1, 0, 0] -> face  
[0, 1, 0] -> airplane  
[0, 0, 1] -> motorbike
### Felmerült problémák
- Több aktivációs függvényt kipróbáltunk (egység, sigmoid, tanh), de csak a sigmoid-ra működött jól  
- Ha túl kevés a kép, akkor nem tudja jól megtanulni és rossz eredményt ad (classification_fewer_images.m)
- Az OnlineLearning függvény sok kép esetén nagyon lassú, nem tudtuk megvárni amíg lefut, csak a classification_fewer_images.m - ben
