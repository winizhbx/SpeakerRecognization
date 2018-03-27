function result = MyHelperTestKNNClassifier(trainedClassifier, featuresTest)
%HelperTestKNNClassifier Test a trained classifier
%
%  Input:
%      trainedClassifier : A trained KNN classifier
%      featuresTest      : A table containing the predictor (Pitch & MFCC)
%                          and response (speaker name) columns
%
%  Output:
%      result : A table containing the name, actual speaker, predicted
%               speaker, and confidence of prediction for each test file
%
% This function HelperTestKNNClassifier is only in support of
% SpeakerIdentificationExample. It may change in a future release.

%   Copyright 2017 The MathWorks, Inc.

% Get groups of rows corresponding to each filename
[Fidx,Filenames] = findgroups(featuresTest.Filename);

total = 0;
correct = 0;
for idx = 1:length(Filenames)
    T = featuresTest(Fidx==idx,2:end);  % Rows that correspond to one file
    predictedLabels = string(predict(trainedClassifier,T(:,1:15))); % Predict
    totalVals = size(predictedLabels,1);

    [predictedLabel, freq] = mode(predictedLabels); % Find most frequently predicted label
    match = freq/totalVals*100;

    actualSpeaker = T.Label{1};
    predictedSpeaker = char(predictedLabel);
    
    if strcmp(actualSpeaker,predictedSpeaker)
        correct = correct + 1;
    end
    total = total + 1;
end
result = correct/total;
end