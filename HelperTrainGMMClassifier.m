function [M, Cov, Weight, speaker_names] = HelperTrainGMMClassifier(features, class_num)
M = {};
Cov = {};
Weight = {};

[speaker_idx, speaker_names] = findgroups(features.Label);
speakers_num = length(speaker_names);

for model_idx=1:speakers_num
	features_train = table2array(features(speaker_idx==model_idx, 2:end-1));
	[M{model_idx},Cov{model_idx},Weight{model_idx}]=gaussmix(features_train,[],[],class_num,'v');
end

end

