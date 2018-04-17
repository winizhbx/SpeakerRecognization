function [features] = ExtractFeatures(datastore)
lenData = length(datastore.Files);
features = cell(lenData,1);
for i = 1:lenData
    [data, info] = read(datastore);
    data = detectVoiced(data, 16*10^3); % remove silence
    features{i} = HelperComputePitchAndMFCC(data(:,1),info);
end
features = vertcat(features{:});
features = rmmissing(features);
%head(features)   % Display the first few rows

end

