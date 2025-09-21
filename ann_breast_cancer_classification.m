% Step 1: Load the UCI Breast Cancer Wisconsin Diagnostic Dataset
filename = 'D:\AIML Advanced NIT Silchar Internship\Matlab Directory\Assignment 2\breast+cancer+wisconsin+diagnostic\wdbc.data';
data = readtable(filename, 'FileType', 'text', 'ReadVariableNames', false);

% Step 2: Extract features and labels
labels = data{:, 2};          % 'M' (malignant) or 'B' (benign)
features = data{:, 3:end};    % 30 numerical features

% Step 3: Convert labels to binary: M = 1, B = 0
binaryLabels = strcmp(labels, 'M');  % Logical array: 1 = malignant, 0 = benign

% Step 4: Normalize features
features = normalize(features);

% Step 5: Prepare for Neural Network
X = features';                      % Features: [features x samples]
Y = full(ind2vec(binaryLabels'+1)); % One-hot encode labels [2 x samples]

% Step 6: Define Neural Network with 3 hidden layers
hiddenLayerSizes = [10 20 30];
net = patternnet(hiddenLayerSizes, 'traingd');  % Use SGD

% Step 7: Set training parameters
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.1;

net.trainParam.epochs = 1000;
net.trainParam.lr = 0.01;
net.trainParam.goal = 1e-5;

% Step 8: Train the Network
[net, tr] = train(net, X, Y);

% Step 9: Get Outputs
outputs = net(X);
predictions = vec2ind(outputs) - 1;  % Convert to 0/1
trueLabels = vec2ind(Y) - 1;

% Step 10: Get indices
trainInd = tr.trainInd;
valInd   = tr.valInd;
testInd  = tr.testInd;

% Step 11: Get subset outputs and targets
trainTargets = Y(:, trainInd);
valTargets   = Y(:, valInd);
testTargets  = Y(:, testInd);

trainOutputs = outputs(:, trainInd);
valOutputs   = outputs(:, valInd);
testOutputs  = outputs(:, testInd);

% Step 12: Plot Confusion Matrices
figure;
subplot(2,2,1);
plotconfusion(trainTargets, trainOutputs);
title('Training Confusion Matrix');

subplot(2,2,2);
plotconfusion(valTargets, valOutputs);
title('Validation Confusion Matrix');

subplot(2,2,3);
plotconfusion(testTargets, testOutputs);
title('Test Confusion Matrix');

subplot(2,2,4);
plotconfusion(Y, outputs);
title('Overall Confusion Matrix');
