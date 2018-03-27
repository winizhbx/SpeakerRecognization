%clc,clear all,close all
dataDir = MyHelperAN4Download;
ads = audioexample.Datastore(dataDir, 'IncludeSubfolders', true,...
    'FileExtensions', '.flac', 'ReadMethod','File',...
    'LabelSource','foldernames');
[trainDatastore, testDatastore]  = splitEachLabel(ads,0.80);
%{
trainDatastore
trainDatastoreCount = countEachLabel(trainDatastore)
testDatastore
testDatastoreCount = countEachLabel(testDatastore)
[sampleTrain, info] = read(trainDatastore);
sound(sampleTrain,info.SampleRate)
reset(trainDatastore);
%}

%% Feature Extraction
features = ExtractFeatures(trainDatastore);
featureVectors = features{:,2:15};

m = mean(featureVectors);
s = std(featureVectors);
features{:,2:15} = (featureVectors-m)./s;
%head(features)   % Display the first few rows

%% Training a Classifier
[trainedClassifier, validationAccuracy, confMatrix] = ...
    HelperTrainKNNClassifier(features);
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
%{
heatmap(trainedClassifier.ClassNames, trainedClassifier.ClassNames, ...
    confMatrix);
title('Confusion Matrix');
%}

%% Testing the Classifier
featuresTest = ExtractFeatures(testDatastore);
featuresTest{:,2:15} = (featuresTest{:,2:15}-m)./s;
%head(featuresTest)   % Display the first few rows

result = HelperTestKNNClassifier(trainedClassifier, featuresTest)