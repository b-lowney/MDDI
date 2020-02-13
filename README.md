# MDDI
Multi-domain Diffraction Identification code 

Authors: Brydon Lowney*, Ivan Lokmer, Gareth Shane O'Brien

Multi-domain Diffraction Identification (MDDI) is a machine learning method for wavefield separation. 
By applying different domain transformations to the data and feeding these into a neural network, seismic events can be identified. 
Here, we have separated the events into three components: diffractions, reflections, and noise. 
There is also a fourth component for identification, which is where diffractions and reflections overlap. 

These functions have been created for MATLAB and requires the following to run:
MATLAB,
Image Processing Toolbox, 
Neural Network Toolbox.

Usage instructions are in the help of each function. To obtain the figures used in the paper the following should be run:

[OUTPUT, NET] = MDDITRAIN('Data/Train/Synth_Train.ASC','Data/Train/Synth_Class.ASC','Data/Train/Synth_Train_Dip.ASC',1,1,1,1,1,1,0);

[PRED] = MDDIPRED('Data/Predict/Synth_Predict.ASC',NET,'Data/Predict/Synth_Predict_Dip.ASC',1,1,1,1,1,1,0);

OUTPUT is the classified image using the trained network NET, PRED is the prediction image on the synthetic data using the trained NET.
Domains used must match in Training and Prediction examples. Training will stop automatically when no longer updating, this took ~150 epochs on our machine. 

Real data examples seen in the paper are not available for confidentiality reasons.
The trained neural network used for the real data examples was trained on a large quantity of real and synthetic data and thus to achieve
a similar result, training on real data is necessary. Actual results may vary dependent on hardware (for speed) and due to the random nature of each trained network.  
