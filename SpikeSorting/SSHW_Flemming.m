%SSHW_Flemming.m
% This script is the code accompanying answers to BIOENG 1586 (Spring 2017)
% Spike Sorting homework assignment. All code is written by Rory Flemming
% unless otherwise specified. Frequently, assisting visuals will be
% commented out to streamline later computations and for debugging.
% Uncomment to generate more descriptive figures.

% STEPS TO SPIKE SORTING
% 1) V_thresh = 3*RMS_Amp set to ID APs. Envelop snippets (-0.4 to 1.6ms) are saved.
% 2) Snippets sampled => 1 dim/sample.
% 3) PCA
% 4) k-means (CV to ID best k)

% Reset
close all; clear all;
% Dataset
load('waveforms.mat');
    %Description:
    % Struct w/ fields:
        % wf = snippets of V
        % stamps = t_AP_i for i = number of action potentials
        
%% 1)Plot 100 of the waveforms, spread throughout the data session.
% Select random 100
rand_ind = randi(length(data.wf),100,1); %create random indexes
rand_wf = data.wf(rand_ind,:)'; % 100 random waveforms

% Get the time axis
ti = -0.4; %start time of sample
tf = 1.6; %end time of sample
n_s = 48; %number of samples
t_s = linspace(ti,tf,n_s); %sample times 

%Plot waveforms ontop of one another
figure
plot(t_s,rand_wf,'LineWidth',1.3)
title('100 Sampled Waveforms','FontWeight','bold','FontSize',16);
xlabel('Sample Time (ms)','FontSize',14);
ylabel('Voltage Activity (no units)','FontSize',14)

%How many neurons would you judge to be present by eye?
% It appears almost impossible to sort neurons in this manner. There is
% large variance in the voltages, which makes for thick groups. It is
% difficult to follow the trajectory of a single neuron in this manner.
% Most events seem to follow a stereotypical event profile. However, it is
% apparent that some events have later recovery profiles (later rises than
% that of the main profile. Based on event amplitude, we see a range of
% approximately 50 units in peak activity. It seems like there are at least
% 2-3 amplitude clusters. Pressed for an answer, I might guess that there
% are 2-5 distinct waveform sources in the dataset.

%% 2)PCA
[coeff,score,latent,tsquare,explained] = pca(data.wf);

% Show waveforms in PCA space by plotting projections onto the first 2
% components against eachother.
figure
scatter(score(rand_ind,1),score(rand_ind,2),'LineWidth',1.3)
title('Principal Components of 100 Waveforms','FontWeight','bold','FontSize',16);
xlabel('PC 1','FontSize',14)
ylabel('PC 2','FontSize',14)


% Get an idea of cluster densities. Use surface() or a colormap. How many
% neurons would you predict to be present?
figure
hist3(score(:,1:2), [20 20])
title('Principal Components of All Waveforms','FontWeight','bold','FontSize',16)
xlabel('PC 1','FontSize',14)
ylabel('PC 2','FontSize',14)
zlabel('Occurances','FontSize',14)

% How many neurons would you say are present in this recording?
% It appears likely that there are two neurons present in the recording.

%% 3)Get clusters - k-means
k_idx = zeros(length(data.wf),5);
for k = 1:5
    k_idx(:,k) = kmeans(score,k);
end
%% 4)Plot 100 waveforms, colored by classification. Did it work?
rand_k = k_idx(rand_ind,:);
for kk = 1:5
    figure
    h = plot(t_s,rand_wf,'LineWidth',2);
    title(sprintf('Classified Waveforms, k = %d',kk),'FontWeight','bold','FontSize',16)
    xlabel('Time (ms)','FontSize',14)
    ylabel('Voltage','FontSize',14)
    
    for m = 1:length(rand_wf);
        if rand_k(m,kk) == 1;
            h(m).Color = 'b';
        elseif rand_k(m,kk) == 2;
            h(m).Color = 'r';
        elseif rand_k(m,kk) == 3;
            h(m).Color = 'g';
        elseif rand_k(m,kk) == 4;
            h(m).Color = 'c';
        else
            h(m).Color = 'y';
        end
    end
end
%% 5) Plot a unit vector from PCA space back into the original space to get the wf.
% Do for first 3 PCs. What does it look like? What intuition does it
% reveal?
figure
plot(t_s,coeff(:,1:3),'LineWidth',2)
legend({'PC 1','PC 2','PC 3'})
title('PC Waveforms','FontWeight','bold','FontSize',16)
xlabel('Time (ms)','FontSize',14)
ylabel('Normalized Voltage','FontSize',14)

% It appears that the different components account for different amplitudes
% and timescales of APs. We see PC 1 is a fast, high amplitude AP, while PC
% 2 is slightly later in time and lower in amplitude. PC 3 also has a
% breadth of delay and lower amplitude, suggesting more scattered spikes at
% a larger distance. PC 3 is likely representive of LFPs, while PC 1 & 2
% may be two nearby neurons.

%% 6) How many dimensions are present in the original data? - eigenspectrum

% An eigenspectrum is a histogram with dimensions on x-axis and total
% variance accounted for on y-axis.
figure
bar(explained,'LineWidth',2)
title('Eigenspectrum of Waveform PCA','FontWeight','bold','FontSize',16)
xlabel('Component','FontSize',14)
ylabel('Variance Eplained (%)','FontSize',14)

figure
plot(explained,'LineWidth',2)
title('Eigenspectrum of Waveform PCA','FontWeight','bold','FontSize',16)
xlabel('Component','FontSize',14)
ylabel('Variance Eplained (%)','FontSize',14)

figure
bar(explained(1:10),'LineWidth',2)
title('Eigenspectrum of Waveform PCA','FontWeight','bold','FontSize',16)
xlabel('Component','FontSize',14)
ylabel('Variance Eplained (%)','FontSize',14)

% How many dimensions does it take to account for 90% of the variance?
count = 0;
var_act = 0;
for i = 1:length(explained)
    if var_act < 90
        count = count + 1;
        var_act = var_act + explained(i);
    else
        break
    end
end
sprintf('It takes %d dimensions to account for 90%% of the variance', count)

% It takes 13 dimensions to account for 90% of the variance.

% Where would you estimate the elbow to be?
% The elbow appears at 4-5 components.

%% 7)Assign rasters to neurons. -- ONLY 2 NEURONS-- Correct Raster is at the bottom

% Choose a stretch of ~10s
exp_dur = data.stamps(end);
rand_ti = randi(floor(exp_dur-10),1);
rand_tf = rand_ti + 10;

% Select time stamps and labels within the interval
win_stamps = nan(length(data.stamps),1);
raster_labels = nan(length(data.stamps),1);

for g = 1:length(win_stamps)
    if data.stamps(g) > rand_ti && data.stamps(g) < rand_tf
        win_stamps(g) = data.stamps(g);
        raster_labels(g) = k_idx(g,2); %Since we have decided that there are 2 neurons
    end
end

% Sort the timestamps by the neuron for input into plotSpikeRaster.m
s1 = [];
s2 = [];

for m = 1:length(win_stamps)
    if ~isnan(win_stamps(m));
        if raster_labels(m) == 1
            s1 = [s1, win_stamps(m)];
        elseif raster_labels(m) == 2
            s2 = [s2, win_stamps(m)];
        end
    end
end
spikes = cell(2,1);
spikes(1,1) = {s1};
spikes(2,1) = {s2};

% Plot spike raster
figure
l = plotSpikeRaster(spikes,'PlotType','vertline');
%Function for plotting spike rasters, developed by Jeffrey Chiaou,
%downloaded from Mathworks forums.
title('Spike Raster','FontWeight','bold','FontSize',16);
xlabel('Time (s)','FontSize',14);
ylabel('Neuron','FontSize',14);
set(gca,'YTick',[1,2],'YTickLabels', {'1','2'})

% Below section is commented out since it doesn't work, but may be useful
% in certain future changes


% % % Plot the rasters during that time, color-coded by cell
% % % values = ones(length(win_stamps),1);
% % % l = stem(win_stamps,values,'filled','.','LineWidth',1);
% % % title('Raster Plot','FontWeight','bold','FontSize',16);
% % % xlabel('Time (s)','Fontsize',14);
% % % set(gca,'yLim',[0,10])
% % % 
% % % NaNcount = 0;
% % % for m = 1:length(win_stamps);
% % %     if raster_labels(m) == 1;
% % %         l(m+NaNcount).Color = 'b';
% % %     elseif raster_labels(m) == 2;
% % %         l(m+NaNcount).Color = 'r';
% % %     elseif raster_labels(m) == 3;
% % %         l(m+NaNcount).Color = 'g';
% % %     elseif raster_labels(m) == 4;
% % %         l(m+NaNcount).Color = 'c';
% % %     elseif raster_labels(m) == 5;
% % %         l(m+NaNcount).Color = 'y';
% % %     elseif raster_labels(m) == NaN;
% % %         NaNcount = NaNcount + 1;
% % %     end
% % % end

%% 8) Inspect ISI histogram
%What predictions do you have about the traits of the ISI histogram?
    % I would predict a refractory period.
    
% Isolate all spike times from each neuron, similar to process above
n1_spikes = zeros(length(find(k_idx(:,3)==1)),1);
n2_spikes = zeros(length(find(k_idx(:,3)==2)),1);
n3_spikes = zeros(length(find(k_idx(:,3)==3)),1);

%Counters
n1c = 0;
n2c = 0;
n3c = 0;

% Compile list of the spikes for each neuron
for i = 1:length(data.stamps)
    
    if k_idx(i,3) == 1 % Use k_idx(_,3) since we assume k=3 neurons
        n1c = n1c + 1;
        n1_spikes(n1c) = data.stamps(i);
    elseif k_idx(i,3) == 2
        n2c = n2c +1;
        n2_spikes(n2c) = data.stamps(i);
    elseif k_idx(i,3) == 3
        n3c = n3c + 1;
        n3_spikes(n3c) = data.stamps(i);
    end
    
end

% Convert spike times to ISI
n1_ISI = zeros(length(n1_spikes)-1,1);
n2_ISI = zeros(length(n2_spikes)-1,1);
n3_ISI = zeros(length(n3_spikes)-1,1);

% Convert spike times to ISI
for i = 1:length(n1_ISI)
    n1_ISI(i) = n1_spikes(i+1) - n1_spikes(i);
end

for i = 1:length(n2_ISI)
    n2_ISI(i) = n2_spikes(i+1) - n2_spikes(i);
end

for i = 1:length(n3_ISI)
    n3_ISI(i) = n3_spikes(i+1) - n3_spikes(i);
end

% Remove outliers/tail end for cleaner plotting
sort(n1_ISI);
sort(n2_ISI);
sort(n3_ISI);


n1_ISI_trim = n1_ISI(1:ceil(length(n1_ISI)*0.85));
n2_ISI_trim = n2_ISI(1:ceil(length(n2_ISI)*0.85));
n3_ISI_trim = n3_ISI(1:ceil(length(n3_ISI)*0.85));


% Create Histograms of ISIs
figure
subplot(3,1,1);
hist(n1_ISI_trim*1000,5000); % Multiplied by 1000 to convert to ms
title('Neuron 1 ISI','FontWeight','bold','FontSize',16)
xlabel('Time(ms)','FontSize',14);
ylabel('Occurances','FontSize',14);
set(gca,'XLim',[0,30])

subplot(3,1,2);
hist(n2_ISI_trim*1000,5000); % Multiplied by 1000 to convert to ms
title('Neuron 2 ISI','FontWeight','bold','FontSize',16)
xlabel('Time(ms)','FontSize',14);
ylabel('Occurances','FontSize',14);
set(gca,'XLim',[0,30])

subplot(3,1,3)
hist(n3_ISI_trim*1000,5000); % Multiplied by 1000 to convert to ms
title('Neuron 3 ISI','FontWeight','bold','FontSize',16)
xlabel('Time(ms)','FontSize',14);
ylabel('Occurances','FontSize',14);
set(gca,'XLim',[0,30])

%Are the ISI histograms consistent with well-isolated neurons?
    %Yes, they have a single, non-zero peak and trail off beyond. There
    %appear to be a number of misclassifications (a spike in spike count
    %occurances nearest to 0). Upon seeing this, I increased the number of
    %k (neurons) until I no longer got reasonable ISI distributions. The
    %limit of k's good fit of ISIs appears to be 3. At k = 4, one of the
    %ISI distributions is a spike near 0 and uniform beyond, which is not
    %indicative of a well-isolate neuron. Overall, the number of
    %misclassifications is relatively small. Using this information, we
    %will go back and redo the raster with 3 neurons.
    
%% Correction #7: Redo spike raster with k = 3
% Choose a stretch of ~10s
exp_dur = data.stamps(end);
rand_ti = randi(floor(exp_dur-10),1);
rand_tf = rand_ti + 10;

% Select time stamps and labels within the interval
win_stamps = nan(length(data.stamps),1);
raster_labels = nan(length(data.stamps),1);

for g = 1:length(win_stamps)
    if data.stamps(g) > rand_ti && data.stamps(g) < rand_tf
        win_stamps(g) = data.stamps(g);
        raster_labels(g) = k_idx(g,3); %Since we have decided that there are 2 neurons
    end
end

% Sort the timestamps by the neuron for input into plotSpikeRaster.m
s1 = [];
s2 = [];
s3 = [];

for m = 1:length(win_stamps)
    if ~isnan(win_stamps(m));
        if raster_labels(m) == 1
            s1 = [s1, win_stamps(m)];
        elseif raster_labels(m) == 2
            s2 = [s2, win_stamps(m)];
        elseif raster_labels(m) == 3
            s3 = [s3, win_stamps(m)];
        end
    end
end
spikes = cell(3,1);
spikes(1,1) = {s1};
spikes(2,1) = {s2};
spikes(3,1) = {s3};

% Plot spike raster
figure
l = plotSpikeRaster(spikes,'PlotType','vertline');
%Function for plotting spike rasters, developed by Jeffrey Chiaou,
%downloaded from Mathworks forums.
title('Spike Raster','FontWeight','bold','FontSize',16);
xlabel('Time (s)','FontSize',14);
ylabel('Neuron','FontSize',14);
set(gca,'YTick',[1,2,3],'YTickLabels', {'1','2','3'})
