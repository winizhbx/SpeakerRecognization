clc,clear all,close all
addpath('voicebox')

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
class_num = 8;
[M, Cov, Weight, speaker_names] = HelperTrainGMMClassifier(features, class_num);

%% Testing the Classifier
featuresTest = ExtractFeatures(testDatastore);
featuresTest{:,2:15} = (featuresTest{:,2:15}-m)./s;

result = HelperTestGMMClassifier(featuresTest, M, Cov, speaker_names, class_num)