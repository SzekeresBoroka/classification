function classification_3classes()
trainTestRatio = 0.5;
lr = 0.0005;

f = @logsig;
gradf = @(x) f(x) .* (1-f(x));

%data
[xTrain, dTrain, xTest, dTest, imgTest] = Load('Faces_easy', 'airplanes', 'motorbikes', trainTestRatio);

fprintf('learning from training data...\n');
[w,E] = OfflineLearning(xTrain, dTrain, f, gradf, lr, @Stop);
%[w,E] = OnlineLearning(xTrain, dTrain, f, gradf, lr, @Stop);
fprintf('E: %.4f\n', E(end));

fprintf('predicting test-data...\n');
y = Predict(xTest, f, w);
y = OutputToClass(y, max(dTest(:)));
Draw(y, dTest, imgTest);

% ----------------------------------------------------------------------------------------------------------------------------------
function Draw(predicted, actual, X)
% predicted
% actual
c = ["face", "airplane", "motorbike"];
close all;
nrows = 6;
ncols = 6;
% predicted = predicted + 1;
% actual = actual + 1;
predicted_index = predicted;
% for i=1:length(predicted)
%     %index of 1 from [1,0,0], [0,1,0], [0,0,1]
%     predicted_index(i,1) = find(predicted(i,:)==1);
% end
for i=1:length(actual)
    %index of 1 from [1,0,0], [0,1,0], [0,0,1]
    actual_index(i,1) = find(actual(i,:)==1);
end

% predicted_index
% actual_index

for i = 1:nrows
    for j = 1:ncols
        k = (i-1)*ncols + j;
        subplot(nrows, ncols, k);
        img = uint8(reshape(X(k,:), 64, 64));
        imshow(img);
        xlabel(c(predicted_index(k)));
    end
end
set(gcf, 'Position', [50,211,560,690]);
MinGui()

figure
confusionchart(c(actual_index), c(predicted_index));
set(gcf, 'Position', [50,0,560,136]);
MinGui();

% ----------------------------------------------------------------------------------------------------------------------------------
function MinGui()
set(gcf(), 'MenuBar', 'none');
set(gcf(), 'MenuBar', 'none');

% ----------------------------------------------------------------------------------------------------------------------------------
function c = OutputToClass(y, alpha)
N = size(y, 1);
c1 = repmat([1 0 0]*alpha, N, 1);
c2 = repmat([0 1 0]*alpha, N, 1);
c3 = repmat([0 0 1]*alpha, N, 1);
d1 = vecnorm(c1-y, 2, 2);
d2 = vecnorm(c2-y, 2, 2);
d3 = vecnorm(c3-y, 2, 2);

[~, c] = min([d1 d2 d3], [], 2);

% ----------------------------------------------------------------------------------------------------------------------------------
function [xTrain, dTrain, xTest, dTest, imgTest] = Load(folder1, folder2, folder3, trainTestRatio)
fprintf('loading images...\n');
[img0, N0] = LoadFolder(folder1);
fprintf('%i images in %s folder\n', N0, folder1);
[img1, N1] = LoadFolder(folder2);
fprintf('%i images in %s folder\n', N1, folder2);
[img2, N2] = LoadFolder(folder3);
fprintf('%i images in %s folder\n', N2, folder3);
%d = [zeros(N0, 1); ones(N1, 1); ones(N2, 1) * 2];
d = [ones(N0, 1)*[1,0,0]; ones(N1,1)*[0,1,0]; ones(N2,1)*[0,0,1]];
img = [img0; img1; img2];
N = N0 + N1 + N2;
p = randperm(N);
d = d(p,:);
X = zscore(img(p,:));
n = round(N* 0.5);
xTrain = X(1:n,:);
xTest = X(n+1:end,:);
%dTrain = (d(1:n, :) == [1 2 3]) * 0.9;
dTrain = d(1:n,:);
dTest = d(n+1:end,:);
imgTest = img(p(n+1:end),:);

% ----------------------------------------------------------------------------------------------------------------------------------
function [img, N] = LoadFolder(folder)
files = dir ([folder '/*.jpg']);
N = length(files);
img = [];
for i = 1:N
    %fprintf('loading %s\n', files(i).name);
    img_i = imread([folder '/' files(i).name]);
    img_i = imresize(img_i, [64, 64]);
    if size(img_i, 3) == 3
        img_i = rgb2gray(img_i);
    end
    img(i,:) = double(img_i(:))';
end

% ----------------------------------------------------------------------------------------------------------------------------------
function z = Stop(E, epoch)
if epoch > 10000
    z = true;
    return;
end
if length(E) < 10
    z = false;
    return;
end

if E(end-9) < E(end) || E(end-9) - E(end) < 1e-3
    z = true;
    return;
end

z = false;
return;

% ----------------------------------------------------------------------------------------------------------------------------------
function y = Predict(x, f, w)
y = zeros(size(x,1), size(w,2));

for i = 1:size(x,1)
    y(i,:) = f((x(i,:)*w));
end