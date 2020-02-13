function [CLASSOUT,NET] = MDDITRAIN(filename,classfile,dipfile,DIP,FX,FK,PREW,CANNY,GRAD,POL)
% Function to train a Multi-Domain Diffraction Identification network on seismic
% images. This function applies a multitude of optional transformations on
% the data in order to better understand the data. 
% [CLASSOUT, NET] = MDDITRAIN(filename,classfile,dipfile,DIP,FX,FK,PREW,CANNY,GRAD,POL)
% 
% CLASSOUT  = The output classification 
% NET       = The trained neural network 
% 
% filename  = The filename of the input data (.ASC format)
% classfile = The filename of the user classification for training (.ASC format)
% dipfile   = The filename of the dip field 
% DIP       = Binary classification of whether to use input dip (0 for no, 1 for yes)
% FX        = Binary classification of whether to use FX domain (0 for no, 1 for yes)
% FK        = Binary classification of whether to use FK domain (0 for no, 1 for yes)
% PREW      = Binary classification of whether to use Prewitt Sobel filter (0 for no, 1 for yes)
% CANNY     = Binary classification of whether to use Canny Sobel filter (0 for no, 1 for yes)
% GRAD      = Binary classification of whether to use image gradient (0 for no, 1 for yes)
% POL       = Binary classification of whether to use polarity (0 for no, 1 for yes)


tic; 

% Define variables

vec = 1;

% Reads in the training classification from the filename 

CLASS = dlmread(classfile);
CLASSVEC = reshape(CLASS,[],1);

% Reshape the CLASS vector for ML

MACHINE   = zeros(length(CLASSVEC),4);

for n = 1:length(CLASSVEC);
    if CLASSVEC(n,1) == 0;
        MACHINE(n,1) = 1;
    end
    if CLASSVEC(n,1) == 1;
        MACHINE(n,2) = 1;
    end
    if CLASSVEC(n,1) == 2;
        MACHINE(n,3) = 1;
    end
    if CLASSVEC(n,1) == 3;
        MACHINE(n,4) = 1;
    end
end


% Reads in the data from the filename

DATA = dlmread(filename);
DATAVEC(:,vec) = reshape(DATA,[],1); vec = vec+1; %Amplitude data
modsize = size(DATA);

% Reads in the dip from the provided dip file (we calculate dips using
% Madagascar software www.ahay.org)

if DIP == 1
    DATADIP        = dlmread(dipfile);
    DATAVEC(:,vec) = reshape(DATADIP,[],1); vec = vec+1; %Dip data
end

% Calculates the FX domain (and maps these points)

if FX == 1
    DATAFFT = zeros(size(DATA));
    for y=1:size(DATA,1);
        for x=1:size(DATA,2);
            DATAFFT(y,x)   = fft(DATA(y,x));
        end
    end
    DATAVEC(:,vec) = reshape(DATAFFT,[],1); vec = vec+1; %FX data
end

% Calculates the FK domain (and maps these points)

if FK == 1
    DATAFFT2 = zeros(size(DATA));
    for y=1:size(DATA,1);
        for x=1:size(DATA,2);
            DATAFFT2(y,x)  = fft(DATA(y,x));
        end
    end
    DATAVEC(:,vec) = reshape(DATAFFT2,[],1); vec = vec+1; %FK data
end

% Calculates the Prewitt Sobel filter

if PREW == 1
    DATAPREW       = edge(DATA, 'Prewitt');
    DATAVEC(:,vec) = reshape(DATAPREW,[],1); vec = vec+1; %Prewitt Sobel data
end

% Calculates the Canny Sobel filter

if CANNY == 1
    DATACANNY      = edge(DATA,'Canny');
    DATAVEC(:,vec) = reshape(DATACANNY,[],1); vec = vec+1; %Canny Sobel data
end

% Calculates the image gradient

if GRAD == 1
    DATAGRAD       = gradient(DATA);
    DATAVEC(:,vec) = reshape(DATAGRAD,[],1); vec = vec+1; %Image Gradient data
end

% Polarity data (checks for changes in polarity i.e. at the apex of
% diffractions)

if POL == 1
    DATAPOS        = (DATA > 0);
    DATANEG        = (DATA < 0);
    DATANEG        = -DATANEG;
    DATAPOL        = DATANEG + DATAPOS;
    DATAVEC(:,vec) = reshape(DATAPOL,[],1); vec = vec+1; %Polarity data
end

% Create and train the model 

x = DATAVEC';
t = MACHINE';

% Training function;
trainFcn = 'trainscg'; %Scaled conjugate gradient backpropagation

% Create the neural network
hiddenLayer1 = 50;
hiddenLayer2 = 50;
% hiddenLayer3 = 50;
hiddenLayerSize(1) = hiddenLayer1;
hiddenLayerSize(2) = hiddenLayer2;
% hiddenLayerSize(3) = hiddenLayer3;
% hiddenLayerSize = 50;
net = patternnet(hiddenLayerSize);

% Divide for training, validation, and testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;

% Train the Network
net.trainParam.showWindow = false;
[net,tr] = train(net,x,t);

% Test the Network 
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
display(performance);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Reshape the data back

y             = y';
CLASSOUTROUND = round(y);

CLASSOUTVEC   = zeros(length(MACHINE),1);

for n = 1:length(MACHINE);
    if CLASSOUTROUND(n,1) == 1;
        CLASSOUTVEC(n,1) = 0;
    end
    if CLASSOUTROUND(n,2) == 1;
        CLASSOUTVEC(n,1) = 1;
    end
    if CLASSOUTROUND(n,3) == 1;
        CLASSOUTVEC(n,1) = 2;
    end
    if CLASSOUTROUND(n,4) == 1;
        CLASSOUTVEC(n,1) = 3;
    end
end

CLASSOUT = reshape(CLASSOUTVEC,modsize(1),modsize(2));
NET      = net;

figure(1);
imagesc(CLASSOUT(1:200,:));
axis off;
set(gcf,'color','w');

toc;

end
