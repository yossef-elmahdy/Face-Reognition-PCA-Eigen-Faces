clc; 

%% Loading the model(PCA) and variables 
load PCA_MODEL_DATA.mat;
[filename, pathname] = uigetfile('*.*', 'Select the Input Image');
filepath = strcat(pathname, filename); 

img = imread(filepath); 
query_image = img; 

img = rgb2gray(img); 
img = imresize(img, [M, N]);
img = double(reshape(img, [1, M*N]));

PCA_img = (img-images_mean)*PCA_Vectors;

%% Computing the L1 Distance with sum(|PCA_Images - PCA_query|)
distance_array = zeros(n, 1);
for i=1:n
    distance_array(i) = sum(abs(PCA_Images(i, :)-PCA_img));
end

%% Best Match Detection 
[result, indx] = min(distance_array);

recognized_image = imread(sprintf('.\\Train\\%d.jpg', indx));

subplot(1, 2, 1) 
imshow(query_image); 
title('Query Face (Test)');
subplot(1, 2, 2) 
imshow(recognized_image);
title('Recognized Face');
