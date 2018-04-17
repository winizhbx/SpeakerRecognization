clc,clear all,close all
addpath('silenceRemoval')
addpath('voicebox')

%dataDir = MyHelperAN4Download; %an4
%dataDir = fullfile('TIMIT','TRAIN','DR1'); %TIMIT
dataDir = fullfile('self','train','en'); %recorded
ads = audioexample.Datastore(dataDir, 'IncludeSubfolders', true,...
    'FileExtensions', '.wav', 'ReadMethod','File',...
    'LabelSource','foldernames');
[trainDatastore, testDatastore] = splitEachLabel(ads,0.80);

%% Feature Extraction
features = ExtractFeatures(trainDatastore);
features(:,2) = [];
featureVectors = features{:,2:15};

m = mean(featureVectors);
s = std(featureVectors);
features{:,2:15} = (featureVectors-m)./s;

featuresTest = ExtractFeatures(testDatastore);
featuresTest(:,2) = [];
featuresTest{:,2:15} = (featuresTest{:,2:15}-m)./s;

%% Training a Classifier
[trainedClassifier, validationAccuracy, confMatrix] = ...
    MyHelperTrainKNNClassifier(features);
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

%% Testing the Classifier
result = MyHelperTestKNNClassifier(trainedClassifier, featuresTest)

%% Training a Classifier
class_num = 8;
[M, Cov, Weight, speaker_names] = HelperTrainGMMClassifier(features, class_num);

%% Testing the Classifier
result = HelperTestGMMClassifier(featuresTest, M, Cov, speaker_names, class_num)