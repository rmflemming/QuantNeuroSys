%CompVision_Flemming.m

%% 1A) Create visual input for the Mach Band Illusion
%create matrix of luminence values
stim = zeros(64,128); 
stim(:,1:32) = 10;%Low
for i = 1:65 %Gradient
    stim(:,32+i) = stim(:,31+i) + 1;
end
stim(:,(32+65):end) = 75; %High

%Plot image
figure
imagesc(stim)
title('Stimulus');
colormap gray


%% 1B) Plot brightness as function of horizontal position
figure
h = plot(1:128,stim(32,:));
h(1).LineWidth = 2;
title('Brightness as a function of horizontal position','FontSize',16);
xlabel('Horizontal Position','FontSize',14);
ylabel('Brightness','FontSize',14)
set(gca,'XLim',[-2,130],'YLim',[5 80]);

%% 1C) Create the receptive field of a retinal ganglion cell: Diff of 2 Gaussians
[X,y] = meshgrid(-2:1:2); % X coordinates and Y coordinates
[TH,R] = cart2pol(X,y); % Polar Coordinates

%%%%%%%%%%%%%%%%%%%%%%%%%% NEEDS EDIT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Egaus = normpdf(R,0,sqrt(2)); % Excitatory, mu = 0, var = 2, 5x5 grid
Igaus = normpdf(R,0,sqrt(6)); % Inhibitory, mu = 0, var = 6, 5x5 grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = 500; %overall strength of the field connectivity
IE = 1; % ratio of inhibition to excitation 
Rfield = S*(Egaus - IE*Igaus); %created receptive field

 figure
 imagesc(Rfield)
 title('Receptive Field');
 colormap gray

%% 1D) Convolve RF and Input to observe first stage of visual system perception
V1 = conv2(stim,Rfield,'valid');
figure
imagesc(V1)
title('Receptive Field x Stimulus Convolution Output');
colormap gray

% Now compare with the stimulus brightness
indx = [1:128;NaN NaN 3:126 NaN NaN]';
data = [stim(32,:);NaN NaN V1(32,:) NaN NaN]';
% 
% figure
% g = plot(indx,data);
% g(1).LineWidth = 2;
% g(2).LineWidth = 2;
% 
% title('Brightness of Stimulus vs Perception','FontSize',16);
% xlabel('Horizontal Position','FontSize',14);
% ylabel('Brightness','FontSize',14)
% legend({'Stimulus','Response'});
% set(gca,'XLim',[-2,130],'YLim',[5 80]);

%% 1 Bonus) Normalize the curves to show the difference
data2(:,1) = data(:,1)./max(data(:,1));
data2(:,2) = data(:,2) ./ max(data(:,2));

figure
g = plot(indx,data2);
g(1).LineWidth = 2;
g(2).LineWidth = 2;

title('Normalized Brightness of Stimulus vs Perception','FontSize',16);
xlabel('Horizontal Position','FontSize',14);
ylabel('Brightness','FontSize',14)
legend({'Stimulus','Response'});
set(gca,'XLim',[-2,130],'YLim',[0.08,1.02]);
%% 2) V1 orientation tuning
%A) Define orientation and spatial frequency
OR = 0;
SF = 0.01;
%B) Coordinate space
[x,y] = meshgrid(-20:1:20);

%C) Create Gabor Function
std_x = 7;
std_y = 17;

gaus2d = @(x,y) exp(-0.5*(((x.^2)/(std_x^2)) + ((y.^2)/(std_y^2))));

Y = -x*sin(OR) + y*cos(OR); %Skew&rotate gausssian
X = x*cos(OR)+y*sin(OR);

A = sin(2*pi*SF*X)/(2*pi*std_x*std_y); %Define modulating sinusoid;
Gabor = A.*gaus2d(X,Y);

%D) Plot the Gabor function

figure
imagesc(Gabor);
colormap gray
title('Gabor Function, OR=0,SF=0.01');
xlabel('Horizontal Position');
ylabel('Vertical Position');

%E) Load Rose and convolve with Gabor
input = imread('rose.jpg');
v1b = conv2(double(input),Gabor,'valid');
figure
subplot(2,3,1)
imagesc(input)
title('Input')
set(gca,'XTickLabel','','YTickLabel','')
colormap gray


subplot(2,3,2)
imagesc(v1b)
title('OR=0,SF=0.01');
set(gca,'XTickLabel','','YTickLabel','')
colormap gray

%F) Play with OR (pi/2,pi/4) and SF (.5,1) and observe output. What do
%these manipulations do to the representation?
OR2 = pi/2; OR3 = pi/4; SF4 = 0.5; SF5 = 1;
X2 = x*cos(OR2) + y*sin(OR2); Y2 = -x*sin(OR2) + y*cos(OR2);
X3 = x*cos(OR3) + y*sin(OR3); Y3 = -x*sin(OR3) + y*cos(OR3);
filter2 = gaus2d(X2,Y2); filter3 = gaus2d(X3,Y3);

A4 = sin(2*pi*SF4*x)/(2*pi*std_x*std_y);
A5 = sin(2*pi*SF5*x)/(2*pi*std_x*std_y);
gaus2d4 = @(x,y) A4.*exp(-((x^2 / (2*std_x^2))+(y^2 / (2*std_y^2))));
gaus2d5 = @(x,y) A5.*exp(-((x^2 / (2*std_x^2))+(y^2 / (2*std_y^2))));

filter4 = gaus2d4(X,Y); filter5 = gaus2d5(X,Y);

v1c = conv2(double(input),filter2,'valid');
v1d = conv2(double(input),filter3,'valid');
v1e = conv2(double(input),filter4,'valid');
v1f = conv2(double(input),filter5,'valid');


subplot(2,3,3)
imagesc(v1c)
title('OR = pi/2, SF = 0.01');
colormap gray
set(gca,'XTickLabel','','YTickLabel','')

subplot(2,3,4)
imagesc(v1d)
title('OR = pi/4, SF = 0.01');
colormap gray
set(gca,'XTickLabel','','YTickLabel','')


subplot(2,3,5)
imagesc(v1e)
title('SF = 0.5, OR = 0');
colormap gray
set(gca,'XTickLabel','','YTickLabel','')


subplot(2,3,6)
imagesc(v1f)
title('SF = 1, OR = 0');
colormap gray
set(gca,'XTickLabel','','YTickLabel','')

%% 3) Gray Spot Illusion
%1) Load and view the illusion image
load('hgrid.mat');
imshow(hgrid)
%2) Construct the receptive field and convolve it with the image
%retina(RF size as frac of image, stdev of kernel = 0.15, image = hgrid)
figure
%retina(.05,0.15,hgrid); % Shows gray constants and back and white edges
%retina(.25,0.15,hgrid); % Shows white bg and black boxes, blurry
%retina(0.005,0.15,hgrid); % Full black, fields too small
%retina(0.1,0.15,hgrid); % Black edges, gray center box, gray x's in junctions where dots go
%retina(0.075,0.15,hgrid); %Shows gray dots in junctions
%retina(0.01,0.15,hgrid); %Gives gray background, only see color at edges
%retina(0.5,0.15,hgrid); %Blurry rounded squares
% ANSWER:
% At a high percentage receptive field coverage, the boxes are resolved
% normally (if a bit blurry, pct > 0.15). However at low percentages of the
% image (pct < 0.1) you see that gray circles/boxes/x's begin appearing in
% the junctions between four boxes. The receptive field size corresponds to
% 1/pct(image), so low pct -> many receptive fields. When you have these
% many receptive fields, they convolve, and average the intensity of the
% surrounding spaces. When the filters are the right size, the black surrounds
% cause the central points to appear closer to gray than white (in the on cells).
% This is a distinct result of surround inhibition.

