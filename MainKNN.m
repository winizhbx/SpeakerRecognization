clc,clear all,close all
dataDir = MyHelperAN4Download;
ads = audioexample.Datastore(dataDir, 'IncludeSubfolders', true,...
    'FileExtensions', '.flac', 'ReadMethod','File',...
    'LabelSource','foldernames');
[trainDatastore, testDatastore]  = splitEachLabel(ads,0.80);

%% Feature Extraction
features = ExtractFeatures(trainDatastore);
featureVectors = features{:,2:15};

m = mean(featureVectors);
s = std(featureVectors);
features{:,2:15} = (featureVectors-m)./s;

%% Training a Classifier
[trainedClassifier, validationAccuracy, confMatrix] = ...
    HelperTrainKNNClassifier(features);
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

%% Testing the Classifier
featuresTest = ExtractFeatures(testDatastore);
featuresTest{:,2:15} = (featuresTest{:,2:15}-m)./s;

result = MyHelperTestKNNClassifier(trainedClassifier, featuresTest)