function [classificationKNN, validationAccuracy, confMatrix] = MyHelperTrainKNNClassifier(trainingData)
% HelperTrainKNNClassifier Train a KNN classifier.
%
%  Input:
%      trainingData: A table containing the predictor and response columns
%
%  Output:
%      trainedClassifier : The trained KNN classifier
%      validationAccuracy: A scalar double containing the validation 
%                          accuracy in percent
%      confMatrix        : The confusion matrix for validation
%
% For example, to train a classifier with the data set T:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = predict(trainedClassifier,T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training.
%
% This function HelperTrainKNNClassifier is only in support of
% SpeakerIdentificationExample. It may change in a future release.

%   Copyright 2017 The MathWorks, Inc.

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = trainingData.Properties.VariableNames;
predictors = inputTable(:, predictorNames(2:end-1));
response = inputTable.Label;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'euclidean', ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'squaredinverse', ...
    'Standardize', false, ...
    'ClassNames', unique(response));

k = 5;
group = (response);
c = cvpartition(group,'KFold',k); % 5-fold stratified cross validation
% Perform cross-validation
partitionedModel = crossval(classificationKNN,'CVPartition',c);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% Compute confusion matrix
validationPredictions = kfoldPredict(partitionedModel);
confMatrix = confusionmat(trainingData.Label, validationPredictions, ...
    'order', classificationKNN.ClassNames);
figure;
confMatrix = diag(sum(confMatrix,2))\confMatrix*100; % convert to percentages


