%RHW_Flemming.m
% This script contains computations for completing BIOENG 1586 Spring 2017
% Homework: Neural and Behavioral Data Analysis. All code is written by
% Rory Flemming, unless otherwise indicated.
%
% Data in this homework comes from neuralData.mat
% This contains neural and behavioral data collected during a delayed
% center-out reaching task performed by a monkey. The monkey fixated at the
% beginning of every trial. Then, a peripheral target box was presented.
% After delay and 'go' signal, the monkey initiated a reach and was allowed
% to move his head. Eight targets at 45 degree intervals make up the
% repretoire of stimuli. Only four recorded neurons are included in this
% dataset.

%%%%%%%%%%%%%%%%%%%%%% PART 1: BEHAVIORAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all;
request = sprintf('You have chosen to run this script! Please wait patiently as it takes a while to plot!')
%% 1) Load data
load('neuralData.mat'); % R struct w/ 29 fields and 1849 entries

%% 2) Find the Unique Target Locations
% Information location:
% R.TrialParams.target1X
% R.TrialParams.target1Y

params = [R.TrialParams]; % Get the trial parameters
targetX = [params.target1X]'; % x positions, col vector
targetY = [params.target1Y]'; % y positions, col vector

% Let's take a quick look at them...
% [targetX(1:10) targetY(1:10)]

% To find unique targets, we will want to find unique combinations of X and
% Y positions. We are at an advantage in that we already know there are 8.

Targets = zeros(8,2); % This will hold the x,y position of our unique targets

% There are many ways to do this, but here is how we will tackle it: We
% will make a matrix of [X,Y]. We will iterate through cols of this matrix
% until we find 8 unique pairs. To determine if a pair is unique, we will
% compare it to our "Targets" matrix. We will create a counter to determine
% the number of unique pairs we find. Once we reach eight, the process will
% terminate.

n_unique = 0;
XY = [targetX targetY];

for i = 1:length(targetX)
    %termination condition
    if n_unique == 8
        break
    end
    
    if isempty(find(Targets(:,1) == XY(i,1), 1)) %Can we find that x value?
        %if no...
        Targets(n_unique+1,:) = XY(i,:);
        n_unique = n_unique + 1;
        
    elseif isempty(find(Targets(:,2) == XY(i,2), 1)) %Can we find the y?
        %if no, it must be a new combo!
        Targets(n_unique+1,:) = XY(i,:);
        n_unique = n_unique + 1;
    end
end

%Let's look at the coordinates of our targets, and just double check that
%they cover all unique X and Y
%Targets, unique(X), unique(Y)

%Quick plot to see what they look like all together
% figure;
% scatter(Targets(:,1),Targets(:,2))

%% 3) Plot Monkey's Hand Position / Trial
% Isolate movement times
%Plan: iterate through each trial. Calculate velocity. Find peak velocity.
%Find the time at which velocity first exceeds 20% of peak, and then when
%it drops below that velocity. We can limit the search within the bounds of
%timeGoCue, ~timePeakVelocity~, and time TargetAcquire.
    %Boundaries
timeGoCue = [R.timeGoCue];
timeTargetAcquire = [R.timeTargetAcquire];
threshold = 0.2; % threshold proportion of max velocity for detecting movement
moveBounds = zeros(length(R),2); %This will hold start and end movement times / trial



for i = 1: length(R) %for each trial
    hhp = R(i).hhp; vhp = R(i).vhp; zhp = R(i).zhp;
    
    %Get velocities
    deltas = zeros(length(hhp)-1,3);
    for kk = 2:length(hhp)
       deltas(kk-1,1) = hhp(kk) - hhp(kk-1); %dx
       deltas(kk-1,2) = vhp(kk) - vhp(kk-1); %dy
        deltas(kk-1,3) = zhp(kk) - zhp(kk-1); %dz
    end
    v = sqrt(deltas(:,1).^2 + deltas(:,2).^2 + deltas(:,3).^2); % 3d velocity
    [maxV, timeMaxV] = max(v(1:timeTargetAcquire(i))); % peak velocity and index for it
    threshV = threshold * maxV; % threshold velocity
    
    % Get movement onset time
    for ii = 0:(timeMaxV - timeGoCue(i)) %limit to timeGoCue -> timeMaxV
        if v(timeGoCue(i)+ii) > threshV
           moveBounds(i,1) = ii + timeGoCue(i);
           break
            
        end
    end
    
    % Get movement end time
    for jj = 0:(timeTargetAcquire(i) - timeMaxV) %limit to timeMaxV -> timeTargetAcquire
        if v(timeMaxV+jj) < threshV
            moveBounds(i,2) = jj + timeMaxV;
            break
        end
    end
    
end

% Trim hand position by movement time
movements = struct([]); %This will hold the trimmed positions

for i = 1: length(R)
    %There is a two-line, faster way to do this, but I am too lazy to
    %recode it... movements.hhp = [movements.hhp; next one]; same for vhp
    new_struct = struct('hhp',[],'vhp',[]);
    if max(R(i).hhp(moveBounds(i,1):moveBounds(i,2))) < 300 %One bad egg
        new_struct.hhp = R(i).hhp(moveBounds(i,1):moveBounds(i,2));
        new_struct.vhp = R(i).vhp(moveBounds(i,1):moveBounds(i,2));
        movements = [movements; new_struct];
    end
end

% Graph hand position traces
    % A fun color palette.
colors = [51 102 255; 102 51 255; 255 51 204; 255 51 102; 255 102 51; 255 204 51; 102 250 102; 51 205 204]/255;
slxns = zeros(length(movements),1); % We will save out the selections for each trial

figure;
for i = 1:length(movements)
    % Color trajectories according towards the reach target:
    % We will want to color the line by the target which was being reached
    % at. We will do this by quickly computing error between selection end
    % point and each target. We will take the target with the smallest error
    % to be the target selected.
    err = (Targets(:,1) - movements(i).hhp(end)).^2 ...
        + (Targets(:,2) - movements(i).vhp(end)).^2; %MSE for each target
    [minErr, slxn] = min(err); %Get the minimum error and the Target chosen
    slxns(i,1) = slxn;
    
    plot([movements(i).hhp],[movements(i).vhp],'Color',colors(slxn,:));
    hold on
end
set(gca,'YLim',[-120 120],'XLim', [-150 150]);
% Plot circles at target locations
scatter(Targets(:,1),Targets(:,2),2000,colors,'filled','MarkerFaceAlpha',0.75,'MarkerEdgeAlpha',0.75);
xlabel('Horizontal Position','fontweight','bold','fontsize',16);
ylabel('Vertical Position','fontweight','bold','fontsize',16);



% 404) Instructions not found

%% 5) Determine RT mean and standard deviations
RTs = moveBounds(:,1) - timeGoCue';
meanRT = mean(RTs);
stdRT = std(RTs);

% For fun, lets plot the RT histogram, I find this to be more informative
%figure;
%hist(RTs,100);
% Since the distribution appears to be Gaussian, and not more like a delta
% dist, we can infer that it is not a single, true RT distribution, but one
% with multiple means. Likely the answer to the next question will be yes.

%% 6) Does RT depend on reach direction? Statistically significant?
% We will want to determine the correct statistical test to determine
% this... Let's see if we can frame the problem a little more thoroughly.
% We have eight targets, and eight RT distributions to go along with each
% target. We could do a permutation test, or maybe an ANOVA, to find out if
% the distributions are different. We may be able to determine another
% machine learning tool for answering this question. I will think on it
% some and come back to it. I absolutely do not want to do the 8c2 post-hoc
% t-tests.



%% 7) What are the mean and standard deviation of the monitor's latency?
    % And what would you estimate as the refresh rate (Hz)?
latency = [R.timeGoCuePHOTO] - timeGoCue;
meanLag = mean(latency);
stdLag = std(latency);

% Again, for fun we will plot the histogram
% figure;
% hist(latency);
% It appears to be a nearly uniform distribution (with a gap in the
% middle?). I guess I would have to guess the refresh rate is 1/meanLag?

%% 8) Plot average eye location by target onto the hand position plot
    % Does the animal tend to look toward the reach target, away, or diff?

% We will plot the average eye location for each trial?? If you say so. I
% would personally try to plot the average eye movement vector for each
% target. Maybe if I have extra time I will attempt that. Let's see how it
% goes...

% Isolate eye movements within the window of movement duration
%R.vep = vertical eye position
%R.hep = horz eye pos

eye_movements = struct([]);

for i = 1:length(R) 
    new_struct = struct('hep',[],'vep',[]);
    if max(R(i).hhp(moveBounds(i,1):moveBounds(i,2))) < 300
        new_struct.hep = R(i).hep(moveBounds(i,1):moveBounds(i,2));
        new_struct.vep = R(i).vep(moveBounds(i,1):moveBounds(i,2));
        eye_movements = [eye_movements; new_struct];
    end
end

%Now we will average each trial
avg_eye = zeros(length(eye_movements),2);

for i = 1:length(eye_movements)
    avg_eye(i,1) = mean(eye_movements(i).hep);
    avg_eye(i,2) = mean(eye_movements(i).vep);
end

% Now we want to plot each average as a point ontop of the reach movements
% We will also color these by target
for i = 1:length(avg_eye)
    hold on;
    scatter(avg_eye(i,1),avg_eye(i,2),[],colors(slxns(i,1),:),'x');
end
hold off

% Yay! We've done it!
%%%%%%%%%%%%%%%%%%%%%%% PART 2: NEURAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1) For one trial, plot rasters for four cells alongside behavior
% Include:
%   Horz/Vert hand pos
%   Horz/Vert eye pos
%   Answer the question

trial_num = randi(length(R),1,1); % select a random trial
timeGo = R(trial_num).timeGoCue;
timeAcquired = R(trial_num).timeTargetAcquire;
timeGoShow = R(trial_num).timeGoCuePHOTO;
neurons = R(trial_num).cells;
hhp = R(trial_num).hhp;
vhp = R(trial_num).vhp;
hep = R(trial_num).hep;
vep = R(trial_num).vep;

spikes = cell(4,1); %We will collect the spike times here
for cells = 1:4
    spikes{cells} = [neurons(cells).spikeTimes]'; % Format spike times for raster
end

% We will use plotSpikeRaster.m for the spike rasters. This code was
% developed by Jeffrey Chiou.
figure;
subplot(3,1,1),plotSpikeRaster(spikes,'PlotType','vertline');
xlabel('Time (ms)','fontweight','bold','fontsize',16);
ylabel('Neuron','fontweight','bold','fontsize',16);
set(gca,'YTick',[1,2,3,4]);
gca();
interval = xlim;
hold on
subplot(3,1,1),plot(repmat(timeGo,100,1),linspace(0,5,100)','r','LineWidth',2);
hold on
subplot(3,1,1),plot(repmat(timeAcquired,100,1),linspace(0,5,100)','g','LineWidth',2);
hold on
subplot(3,1,1),plot(repmat(timeGoShow,100,1),linspace(0,5,100)','b','LineWidth',2);
subplot(3,1,2),plot(hhp),hold on, subplot(3,1,2),plot(vhp),...
    ylabel('Hand Position','fontweight','bold','fontsize',16),...
    legend({'HHP','VHP'});
set(gca,'XLim',interval);

subplot(3,1,3),plot(hep),hold on, subplot(3,1,3),plot(vep),...
    ylabel('Eye Position','fontweight','bold','fontsize',16),...
    legend({'HEP','VEP'});
set(gca,'XLim',interval);

%% 2) For rightward reaches, plot the PSTH for the four cells
% Use the time -300 to +600 ms from cue onset
% We will use the target in the upper right corner, (86,50)
timeStart = [];
upperRightIndex = [];
for i = 1:length(R)
    if XY(i,1) == 86 && XY(i,2) == 50 %Only for upper right target
        timeStart = [timeStart; R(i).timeGoCuePHOTO - 300];
        upperRightIndex = [upperRightIndex; i];
    end
end
% Want binary spike inputs. Create matrix with rows = trials, cols = time
% in ms. Make one for each cell
spikes1 = zeros(length(upperRightIndex),900);
spikes2 = zeros(length(upperRightIndex),900);
spikes3 = zeros(length(upperRightIndex),900);
spikes4 = zeros(length(upperRightIndex),900);

% Get spike times during the 900 ms window. allign data so cue appears at
% the 300th col. For each spike time, put a 1 within the corresponding bin.
cell1 = struct([]);
cell2 = struct([]);
cell3 = struct([]);
cell4 = struct([]);

for i = 1:length(upperRightIndex) %For each right target
    %Look at the spike times for it and assign 1s in appropriate bin
    ind = upperRightIndex(i);
    new_struct = [R(ind).cells];
%     if isempty(upperRightCells)
%         upperRightCells.cell1 = new_struct(1);
%         upperRightCells.cell2 = new_struct(2);
%         upperRightCells.cell3 = new_struct(3);
%         upperRightCells.cell4 = new_struct(4);
%     end
    cell1 = [cell1; new_struct(1)];
    cell2 = [cell2; new_struct(2)];
    cell3 = [cell3; new_struct(3)];
    cell4 = [cell4; new_struct(4)];
    %upperRightCells = [R(upperRightIndex).cells];
end
% Now that we have the right trials, we will transform the spike times into
% the binary data
for i = 1:length(upperRightIndex)
    spike1times = cell1(i).spikeTimes - timeStart(i); % Get the spike times
    spike2times = cell2(i).spikeTimes - timeStart(i); % Shifted by the cue
    spike3times = cell3(i).spikeTimes - timeStart(i); % Onset
    spike4times = cell4(i).spikeTimes - timeStart(i);
    
    %Now we have to select only spike times within our interval
    spike1times = spike1times(spike1times > -300);
    spike2times = spike2times(spike2times > -300);
    spike3times = spike3times(spike3times > -300);
    spike4times = spike4times(spike4times > -300);
    
    spike1times = spike1times(spike1times < 599.5);
    spike2times = spike2times(spike2times < 599.5);
    spike3times = spike3times(spike3times < 599.5);
    spike4times = spike4times(spike4times < 599.5);
    
    spike1times = round(spike1times);
    spike2times = round(spike2times);
    spike3times = round(spike3times);
    spike4times = round(spike4times);
    
    % Now we will transform the spike times into 1's
    for ii = 1:length(spike1times)
        spikes1(i,spike1times(ii)+301) = spikes1(i,spike1times(ii)+301) + 1;
    end
    for ii = 1:length(spike2times)
        spikes2(i,spike2times(ii)+ 301) = spikes2(i,spike2times(ii)+ 301) + 1;
    end
    for ii = 1:length(spike3times)
        spikes3(i,spike3times(ii)+ 301) = spikes3(i,spike3times(ii)+ 301) + 1;
    end
    for ii = 1:length(spike4times)
        spikes4(i,spike4times(ii) + 301) = spikes4(i,spike4times(ii) + 301) + 1;
    end
    
end

%Convolve spike times with a gaussian kernel.
gaus = fspecial('gaussian',5,0.5);
fr1 = mean(spikes1)*1000;
fr2 = mean(spikes2)*1000;
fr3 = mean(spikes3)*1000;
fr4 = mean(spikes4)*1000;
cell1out = conv2(fr1,gaus);
cell2out = conv2(fr2,gaus);
cell3out = conv2(fr3,gaus);
cell4out = conv2(fr4,gaus);

%Label axes. Make sure y axis units are correct. Data are binned at 1 ms
%intervals.
figure;
subplot(4,1,1), bar(cell1out,'k');
set(gca,'XLim',[0 900],'XTickLabels',-300:100:600);
subplot(4,1,2), bar(cell2out,'k');
set(gca,'XLim',[0 900],'XTickLabels',-300:100:600);
subplot(4,1,3), bar(cell3out,'k');
set(gca,'XLim',[0 900],'XTickLabels',-300:100:600);
subplot(4,1,4), bar(cell4out,'k');
xlabel('Time relative to Cue(ms)','fontweight','bold','fontsize',16);
ylabel('Spikes/s','fontweight','bold','fontsize',16);
set(gca,'XLim',[0 900],'XTickLabels',-300:100:600);


%% 3) For each of the four cells, plot a tuning curve
%Can unwrap the target array, to make it cartesian
%To compute one point: add up all spike times during the epoch from 100 ms
%after cue until 600ms after. Then, take average
[TargAng,~] = car2pol(Target(:,1),Target(:,2))

% Plot the avg fr as a function of target location. Y-axis = spikes/s

% Fit the data with a cosine

% Using the fit, find the preferred directions for the four cells. What are
% they?




%%%%%%%%%%%%%%%%%%%%%%%%%%%%% THE END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%