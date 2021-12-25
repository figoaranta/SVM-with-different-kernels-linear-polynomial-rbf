close all;
clear all;

filename = 'AirQualityUCI.xlsx';
[data,header] = xlsread(filename);

header(3) = [];
header(4) = [];

% <--Data Processing(date, time, mat, format)-->
dateTimeData = data(:,(1:2));
dateTimeData = sum(dateTimeData,2);
dateTimeData = datestr(dateTimeData+datenum('30-Dec-1899'));

monthVec = [];
hourVec = [];

for i = 1:length(dateTimeData)
    splitData = split(dateTimeData(i,:));
    rawDate = splitData(1);
    splitRawDate = split(rawDate,'-');
    monthData = splitRawDate(2);
    monthVec = [monthVec,monthData];

    rawTime = splitData(2);
    splitRawTime = split(rawTime,':');
    hour = splitRawTime(1);
    hourVec = [hourVec,hour];
end

monthVec = month(datetime(monthVec,'InputFormat','MMM'));
monthVec = monthVec';
hourVec = str2double(hourVec);
hourVec =hourVec';

training_vectors = [monthVec hourVec data(:,4) data(:,6) data(:,7) data(:,8) data(:,9) data(:,10) data(:,11) data(:,12) data(:,13) data(:,14) data(:,15)];
training_labels = data(:,3);
save('airQuality.mat','training_vectors','training_labels');


% <--Remove Nan, normalise data, finding pearson correlation-->
load("airQuality.mat");

% Remove any rows containing Nan
training_vectors(training_vectors == -200) = NaN;
training_labels(training_labels == -200) = NaN;

training_labels(any(isnan(training_vectors), 2), :) = [];
training_vectors(any(isnan(training_vectors), 2), :) = [];

training_vectors(any(isnan(training_labels), 2), :) = [];
training_labels(any(isnan(training_labels), 2), :) = [];

y = training_labels;
mean_y = mean(y);

pearson_coefficient = [];
for i=1:size(training_vectors,2)
    max_value = (max(training_vectors(:,i)));
    if max_value >= 1
        training_vectors(:,i) = training_vectors(:,i)/max_value;
    end
    x = training_vectors(:,i);
    mean_x = mean(x);
    r = sum((x-mean_x).*(y-mean_y))/ sqrt(sum((x - mean_x).^2).* sum( (y - mean_y).^2));
    pearson_coefficient = [pearson_coefficient,r];
    disp("Pearson Correlation of Feature "+header(i)+" and label (CO(GT)) is: "+r);
end

[higest_r,highest_r_feature] = max(pearson_coefficient);
[lowest_r,lowest_r_feature] = min(pearson_coefficient);

X = [training_vectors(:,highest_r_feature) training_vectors(:,lowest_r_feature)];

% Normalize y
y_max_value =(max(y));
y = y/y_max_value;

% Shuffle Data
% concatData = [X,y];
% shuffledData = concatData(randperm(size(concatData, 1)), :);
% X = shuffledData(:,1:2);
% y = shuffledData(:,3);

save("pearson_air_quality","X","y");











% <-- To check if pearson's formula is correct -->
% x = [17 13 12 15 16 14 16 16 18 19];
% y = [94 73 59 80 93 85 66 79 77 91];
% 
% mean_x = mean(x);
% mean_y = mean(y);
% 
% r = sum((x-mean_x).*(y-mean_y))/ sqrt(sum((x - mean_x).^2).* sum( (y - mean_y).^2)) ;



