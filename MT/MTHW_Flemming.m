%MTHW.m
%% 1) Construct a psychometric curve for the monkey's behavior.
load('MTdata.mat');
beh = MTdata(:,3,:); %isolate monkey behavioral data
pctBeh = zeros(1,6); %create matrix to hold percent correct responses for each
                        %coherence level
for i = 1:6
    pctBeh(i) = mean(beh(:,1,i)); %fill matrix with pct correct for each coh
end

%Let's do a quick check to make sure things look alright
figure
scatter(coherence,pctBeh,'LineWidth',2)
title('Psychometric fit of Behavior','Fontsize',16)
xlabel('Coherence','Fontsize',14);
ylabel('Proportion correct','Fontsize',14);
set(gca,'xscale','log')

%Looks good, let's close out for now

Weibull = @(coefficients,coherence) 1 - 0.5*exp(-(coherence./coefficients(1)).^coefficients(2)); %Weibull function

Beta0 = [0.1,1.1];%initial parameter estimates, alpha,beta
%coherence(1) = 0.0001; %Weibull functions will not accept 0 value

behfit = nlinfit(coherence,pctBeh,Weibull,Beta0);

p = Weibull(behfit,0:0.01:1);

hold on
plot(0:.01:1,p,'LineWidth',2);
set(gca,'xlim',[0.01;1.01]);

%% 2) Construct a neurometric curve from the pair of neurons, implement ROC

% For each coherence, slide criterion from (0,100), and calculate the %
% null and preferred distributions greater than that criterion
% Hint : use commands sort(),find(), and sum()

LFRs = sort(MTdata(:,1,:)); %trials sorted by firing rate for each coh
RFRs = sort(MTdata(:,2,:));

null_above = zeros(101,6); %store output of how many trials exceed criterion
pref_above = zeros(101,6);

for criterion = 0:100
    for cohr = 1:6
        null_ind = find(LFRs(:,1,cohr) > criterion);
        pref_ind = find(RFRs(:,1,cohr) > criterion);
        if size(null_ind) > 0
            null_above(criterion+1,cohr) = (101 - null_ind(1))/100;
        else
            null_above(criterion+1,cohr) = 0;
        end
        
        if size(pref_ind) > 0
            pref_above(criterion+1,cohr) = (101 - pref_ind(1))/100;
        else
            pref_above(criterion+1,cohr) = 0;
        end
    end
end

%Plot those values against eachother
figure;
plot(null_above,pref_above,'LineWidth',2)...
    ,title('ROC Analysis: Pref-Null Response Distributions'...
    ,'FontWeight','bold','Fontsize',16)...
    ,xlabel('P(null > crit)','Fontsize',14)...
    ,ylabel('P(pref > crit)','Fontsize',14)...
    ,legend({'0','4%','8%','16%','32%','64%'});


%Calculate area under the curve. Hint: trapz() . Normalize as fraction of
%whole area
neural_acc = zeros(1,6);
for cohr = 1:6
    neural_acc(1,cohr) = -trapz(null_above(:,cohr),pref_above(:,cohr));
end
%Fit these values to a Weibull function to get neurometric curve
figure
scatter(coherence,neural_acc,'LineWidth',2)
title('Neurometric Data','Fontsize',16)
xlabel('Coherence (%)','Fontsize',14);
ylabel('Proportion correct','Fontsize',14);
set(gca,'xscale','log')

neurofit = nlinfit(coherence,neural_acc,Weibull,Beta0);

p2 = Weibull(neurofit,0:.01:1);

hold on
plot(0:.01:1,p2,'LineWidth',2);
set(gca,'xlim',[0.01;1.01]);

%% 3) Calculate Choice Probability
load('MTdata.mat');

lnlr = zeros(101,6);
lnrr = zeros(101,6);
rnlr = zeros(101,6);
rnrr = zeros(101,6);

for neuron = 1:2
    for coh = 1:6;
        
        %Get the left & right choices
        lefts = find(MTdata(:,3,coh) == 0);
        rights = find(MTdata(:,3,coh) == 1);
        
        %Get the neurons' responses for these choices
        if neuron ==1
            pref_choices = sort(MTdata(lefts,neuron,coh));
            null_choices = sort(MTdata(rights,neuron,coh));
            in = 1; %for saving results
        else
            pref_choices = sort(MTdata(rights,neuron,coh));
            null_choices = sort(MTdata(lefts,neuron,coh));
            in = 3;
        end
        null_above = zeros(101,1);
        pref_above = zeros(101,1);
        %Do sliding criterion to get the ROC value
        for crit = 0:100
            null_ind = find(null_choices > crit);
            pref_ind = find(pref_choices > crit);
            
            if size(null_ind)>0
                null_above(crit+1,1) = (101 - null_ind(1))/100;
            else
                null_above(crit+1,1) = 0;
            end
            
            if size(pref_ind) > 0
                pref_above(crit+1,1) = (101 - pref_ind(1))/100;
            else
                pref_above(crit+1,1) = 0;
            end
        end
        %Save ROC output values for this coherence
        if neuron == 1
            lnlr(:,coh) = pref_above;
            lnrr(:,coh) = null_above;
        else
            rnrr(:,coh) = pref_above;
            rnlr(:,coh) = null_above;
        end
    end
end

%Make the ROC plots

figure;
plot(lnrr,lnlr,'LineWidth',2)...
    ,title('ROC: Left-preferred Neuron'...
    ,'FontWeight','bold','Fontsize',16)...
    ,xlabel('P(FR\_right > crit)','Fontsize',14)...
    ,ylabel('P(FR\_left > crit)','Fontsize',14)...
    ,legend({'0','4%','8%','16%','32%','64%'});

figure;
plot(rnlr,rnrr,'LineWidth',2)...
    ,title('ROC: Right-preferred Neuron'...
    ,'FontWeight','bold','Fontsize',16)...
    ,xlabel('P(FR\_left > crit)','Fontsize',14)...
    ,ylabel('P(FR\_right > crit)','Fontsize',14)...
    ,legend({'0','4%','8%','16%','32%','64%'});


% Take area under ROC curve to get choice probability points
left_acc = zeros(6,1);
right_acc = zeros(6,1);

for coh = 1:6
   left_acc(coh,1) = -trapz(lnrr(:,coh),lnlr(:,coh));
   right_acc(coh,1) = -trapz(rnlr(:,coh),rnrr(:,coh));
end
IndividualChoiceProbability = [left_acc;right_acc];

choice_probability = mean(IndividualChoiceProbability);
cc = repmat(choice_probability,1,100);
ii = linspace(0,3,100);

figure
h = histogram(IndividualChoiceProbability,'BinMethod','sturges');
title('Distribution of Choice Probabilities','FontWeight','bold','FontSize',16);
xlabel('Choice Probability','FontSize',16);
ylabel('Occurances','FontSize',16);
hold on
plot(cc,ii,'LineWidth',3);
legend({'Individual Choice Probabilities','Mean'})
%% 4) Does the animal have an internal bias that affects his choices?
% If so, what do you estimate it to be?
% Yes, 10% bias leftwards.

