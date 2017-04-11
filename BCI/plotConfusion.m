function [ConfuMat] = plotConfusion(Results)
% Plot the confusion matrix: decoded target versus actual target.

[numtarg, numrep] = size(Results);

% Pull out the actual target locations that
% appear in Results
ats = [];
dts = [];  % decoded targs

for j = 1:numtarg
  ats = [ats, Results(j,:).actualTarg];
  dts = [dts, Results(j,:).decodedTarg];
end

ats = unique(ats);

dts = unique(dts);

% Build a big matrix
ConfuMat = repmat(0, [numtarg, numtarg]);


for targInd = 1:numtarg
  for repInd = 1:numrep
    dt = Results(targInd, repInd).decodedTarg;
    dtIndex = find(dt == dts);
    
    x = targInd;
    y = dtIndex;
    
    ConfuMat(x, y) = ConfuMat(x, y) + 1; 
  end
end


%But, ultimately, get it so that 1,1 is in LL corner, not UL corner
% Note that fliplr and flipud probably could have done this easier.
yflipConfuMat = [];
for index = 1:size(ConfuMat,2)
  confuCol = ConfuMat(:,index);
  yflipConfuMat(:,index) = confuCol(end:-1:1);
end
figure
imagesc(yflipConfuMat);

%%% X axis
set(gca,'XTick',[1:numtarg]);

set(gca,'XTickLabel',ats);
xlabel('Actual target');
%%%

%%% Y axis
set(gca,'YTick',[1:numtarg]);

set(gca,'YTickLabel', dts(end:-1:1));
ylabel('Decoded target');
%%%


colormap(1-gray)
%colormap(autumn)

axis square
