function [CLASSOUT] = MDDIPRED(filename,NET,dipfile,DIP,FX,FK,PREW,CANNY,GRAD,POL)
% Function to predict using a Multi-Domain Diffraction Identification network on seismic
% images. This function applies a multitude of optional transformations on
% the data in order to better understand the data. 
% [CLASSOUT] = MDDIPRED(filename,NET,classfile,dipfile,DIP,FX,FK,PREW,CANNY,GRAD,POL)
% 
% CLASSOUT  = The output classification 
% 
% filename  = The filename of the input data (.ASC format)
% NET       = The trained neural network 
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

% Apply neural network 

y = NET(x);

% Reshape the data back

y             = y';
CLASSOUTROUND = round(y);

CLASSOUTVEC   = zeros(length(DATAVEC),1);

for n = 1:length(DATAVEC);
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

figure(1);
imagesc(CLASSOUT(1:200,:));
axis off;
set(gcf,'color','w');

toc;

end
