%BCIhw_Flemming.m
% This script is code to accompany/complete BIOENG 1586 Homework: Neural
% Prosthetic Decoding, spring 2017. All code is written by Rory Flemming
% unless otherwise indicated.

%% Part 1: Continuous Decoding - Linear Filter
% Load data
load('continuous1.mat'); % Courtesy Nicholas Hatsopoulos, U. Chicago
    % Dataset contains two variables:
    %   kin: position of hand (x,y). rec over 1 hr, 70ms sampling times
    %   rate: avg firing rate (70ms bins) activity of 42 M1 neurons
    
%% 1) Visualize Behavioral Data
% Task: random target tracking
% use the function comet() to replay behavioral performance
%comet(kin(:,1),kin(:,2));

%Generate a figure showing the behavioral data. Your choice, make it good
% Here's my plan of attack:
%   We're going to make a 2D comet visualization of only ~7 seconds of
%   the data. So, we will select a random interval of 100 bins, and use
%   those to plot the reach trajectories.
t_start = randi(length(kin)-100,1,1);
t_end = t_start + 100;
figure;
comet(kin(t_start:t_end,1),kin(t_start:t_end,2));
xlabel('Horizontal Position','fontweight','bold','fontsize',16); %These actually arive late
ylabel('Vertical Position','fontweight','bold','fontsize',16);
title('Monkey Hand movements, 7 seconds','fontweight','bold','fontsize',16);

%% 2) Compute Linear Filter Coefficients
% Lin filter: Y = XA
%   Y - 2D kinematics
%   X - neural activity
%   A - coefficients
% Make sure to append a vector of 1's to X to allow for the intercept term
X = [rate, ones(length(rate),1)];

%We will use one time lag to determine cursor output from neural data. Do
%this by shifting behavioral data. Do this by 2 time bins (140ms), which
%is close to latency from M1 to hand movement.
lag = 140; %lag in ms
shift = round(lag/70); % calculate number of bins needed to shift
Y = kin(shift+1:end,:); % shifted kinetics
X = X(1:end - shift,:); %take out trials without a decoding

% Solve for A to build decoder
A = mldivide(Y,X); % I define the decoder in continuous_decoder.m

%% 3) Observe and Quantify Results
% Test decoder performance by multiplying the neural data by the
% coefficient matrix A.
train_test = continuous_decoder(A,X);

% Plot the results in an informative format

%Small Picture
t_start = randi(length(kin)-1000-shift,1,1);
t_end = t_start + 1000;
figure;
comet(kin(t_start:t_end,:));
hold on
comet(train_test(t_start+shift:t_end+shift,:));
xlabel('Horizontal Position','fontweight','bold','fontsize',16); %These actually arive late
ylabel('Vertical Position','fontweight','bold','fontsize',16);
title('Monkey Hand movements, 7 seconds','fontweight','bold','fontsize',16);

%Big picture

% Does the decoder do better at reconstructing the small details of the
% movement or the "big picture"?

% Quantify performance somehow. Consider measuring the MSE between actual
% and reconstructed reaches. 

%% 4) Load continuous2 and attempt to reconstruct new kinematic data
% Load data
load('continuous2.mat');

% Predict outputs

% Plot the reconstructions

% Quantify the fit w/ correlation and MSE

% Compare to train/test on continuous.mat


%% 5) Try various lags
% 0, 70, 210 ms?

% Which gives best predicitions?

% What if you use anti-causal time lag?

% If you want to build a full model that allows multiple time lags, it's
% a simple adjustment to the data matrix = include columns for each time
% lag.  Does that yield better performance? How many time lages give best
% results? Do you think including more time lags always yields better
% reconstructions?


%% Part 2: Discrete Decoding - Bayes Clf
% Since variability in neural responses is generally assumed Poisson, we
% only need to estimate a single parameter, lambda, for each neuron and
% target. It can be estimated easily as the mean spike count observed over
% repetitions of a reach.
% Quantifying this decoder will be simple: generate a "confusion matrix"
% that plots actual target location vs decoded target location. This allows
% you to see how often the decorder is correct, and when errors are made,
% you can see if it's off by much.

% 1) Load data
load('spikeCounts.mat'); % Courtesy Krishna Shenoy, Stanford University
    % 3D matrix of firing rates w/ dims: [repetitions,neurons,targets]
    
% Get a feel for the tuning of the cells by plotting the avg spike count
% for the 5 targets for a handful of cells. We will 10 cells.
datasize = size(SpikeCounts);
rand_ind = randi(datasize(2),1,10);
tuning = zeros(10,5); % rows- neurons, cols - targets
for i = 1:5
    tuning(:,i) = mean(squeeze(SpikeCounts(:,rand_ind,i)));
end
figure; 
plot(tuning','linewidth',2);
xlabel('Target','fontweight','bold','fontsize',16);
ylabel('Mean Firing Rate','fontweight','bold','fontsize',16);
title('Tuning of Ten Neurons','fontweight','bold','fontsize',16);
set(gca,'xtick',1:5,'xticklabel',{'1','2','3','4','5'});

%% 2,3) Build the decoder/ LOOCV


% Reformat the data for table
features = reshape(SpikeCounts,[16*5,91]) + 10E-3* randn(1,1); % Some smoothing and noise added
labels = [ones(16,1);repmat(2,16,1);repmat(3,16,1);repmat(4,16,1);repmat(5,16,1)];

removedX = features;
Results = [labels zeros(80,1)];

for i = 1: 80
% Remove the row of data for training
if i == 1
    feats = features(2:end,:);
    label = labels(2:end,:);
elseif i == 2
    feats = [features(1,:); features(3:end,:)];
    label = [labels(1,:); labels(3:end,:)];
elseif i == 80
    feats = features(1:end-1,:);
    label = labels(1:end-1,:);
else
    feats = [features(1:i-1,:); features(i+1:end,:)];
    label = [labels(1:i-1,:); labels(i+1:end,:)];
end

% Make data into table
datastruct = array2table(feats);
label = array2table(label);
datastruct = [datastruct label];

% A. Train the decoder using all the trials but the one set aside.
% Essentially, comput mean spike counts for each neuron on each target.
discrete_decoder = fitcnb(datastruct,label);
% B. Have the decoder predict the removed label
Results(i,2) = predict(discrete_decoder,removedX(i,:));
end

%% Plot the Confusion Matrix
% Can use 'plotconfusion.m' or make your own. Build a 2D histogram with
% actual target on x axis, and decoded target on y axis. Values are the
% number of occurances of each combination of actual versus decoded. Plot
% the histogram as a colormap.
OutMat = zeros(5);
for y = 1:5
    for x = 1:5
        count = 0;
        for i = 1:80
            if Results(i,1) == x && Results(i,2) == y
                count = count + 1;
            end
        end
        OutMat(x,y) = count;
    end
end
yflipConfuMat = [];
for index = 1:size(OutMat,2)
  confuCol = OutMat(:,index);
  yflipConfuMat(:,index) = confuCol(end:-1:1);
end
figure;
imagesc(yflipConfuMat);
colorbar;
xlabel('Actual Target','fontweight','bold','fontsize',16);
ylabel('Decoded Target','fontweight','bold','fontsize',16);
title('Confusion Matrix','fontweight','bold','fontsize',16);
set(gca,'XLim', [0.5 5.5],'xtick',1:5,'xticklabels',{'1','2','3','4','5'}...
    ,'YLim',[0.5 5.5],'ytick',1:5,'yticklabels',{'5','4','3','2','1'});
% What is the decoder's overall percent correct? What are your thoughts on
% how to improve the endpoint decoder's performance?
