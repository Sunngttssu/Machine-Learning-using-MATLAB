% 1. Load IRIS Dataset from CSV
filename = 'D:\AIML Advanced NIT Silchar Internship\Matlab Directory\Assignment 2\IRIS Flower Dataset\IRIS.csv'; 
data = readtable(filename);

% Extract numeric features (5 features in total but we take 4 feautures as
% the last column signifies the "species")
features = data{:, {'sepal_length', 'sepal_width', 'petal_length', 'petal_width'}};

% Extract class labels
labels = data.species;
[label_numeric, label_names] = grp2idx(labels);

% 2. Extract Statistical Features (Mean, Min, Max, Std)
num_samples = size(features, 1);
stat_features = zeros(num_samples, 4);

for i = 1:num_samples
    sample = features(i, :);
    stat_features(i, 1) = mean(sample);
    stat_features(i, 2) = min(sample);
    stat_features(i, 3) = max(sample);
    stat_features(i, 4) = std(sample);
end

% 3. Train/Test Split
cv = cvpartition(label_numeric, 'HoldOut', 0.3); % 70% train, 30% test
trainX = stat_features(training(cv), :);
testX = stat_features(test(cv), :);
trainY = label_numeric(training(cv));
testY = label_numeric(test(cv));

% 4. Train Classifier (SVM)
model = fitcecoc(trainX, trainY); % Multi-class SVM

% 5. Predict
predictedY = predict(model, testX);

% 6. Accuracy
accuracy = sum(predictedY == testY) / numel(testY) * 100;
fprintf('Classification Accuracy: %.2f%%\n', accuracy);

% 7. Confusion Matrix
figure;
confusionchart(testY, predictedY, 'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title('Confusion Matrix (Statistical Features)');

% 8. 3D Feature Visualization
figure;
hold on;
colors = lines(length(label_names));  % Automatically pick distinct colors

for i = 1:length(label_names)
    idx = (label_numeric == i);
    scatter3(stat_features(idx,1), stat_features(idx,2), stat_features(idx,4), ...
        50, colors(i,:), 'filled');
end

xlabel('Mean');
ylabel('Min');
zlabel('Std Dev');
grid on;
title('IRIS Dataset - Statistical Feature Space');
legend(label_names, 'Location', 'best');
hold off;
