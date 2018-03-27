function result = HelperTestGMMClassifier(featuresTest, M, Cov, speaker_names, class_num)
[Fidx,Filenames] = findgroups(featuresTest.Filename);

speakers_num = length(speaker_names);
total = 0;
correct = 0;
for idx = 1:length(Filenames)
    max = -10e10;
    T = featuresTest(Fidx==idx,2:end);  % Rows that correspond to one file
    for model_idx = 1:speakers_num
        accu = 0;
        for i = 1:size(T,1)
            accu = accu + log(gaussianND(table2array(T(i,1:14)),M{model_idx},Cov{model_idx},class_num));
        end
        if accu > max
            max = accu;
            index = model_idx;
        end
    end
    
    actualSpeaker = T.Label{1};
    predictedSpeaker = speaker_names{index};
    if strcmp(actualSpeaker, predictedSpeaker)
        correct = correct + 1;
    end
    total = total + 1;
end
result = correct/total;

end

