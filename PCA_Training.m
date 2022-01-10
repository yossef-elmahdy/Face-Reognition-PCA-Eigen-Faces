clear all, close all, clc;
%% Initialization
n = input('Enter No. of Images to train: ');
L = input('Enter No. of eigen vectors (singular values) we need: ');
M = 100; N = 90;                                                    % Sizes of the compressed image (N*M)

resized_images = zeros(n, M*N);                                     % Matrix of compressed image (n * (M*N)) --> e.g 300 * 9000
PCA_Images = zeros(n, L);                                           % Vector of PCA eigen vectors (singular vectors) (n*L)

%% Loading the training images in TrainingDB folder
for count=1:n
    I = imread(sprintf('./Train/%d.jpg', count));
    I = rgb2gray(I);                                                % Grayscale for easy exctraction
    I = imresize(I, [M, N]);                                        % Resize the image to M*N
    resized_images(count, :) = double(reshape(I, [1, M*N]));        % Reshape the image to 1*(M*N)
end

resized_images_copy = resized_images;

%% PCA Steps
images_mean = mean(resized_images);

for i=1:n
    resized_images(i, :) = resized_images(i, :) - images_mean;           %Center the image around its mean
end

Q = (resized_images'*resized_images)/(n-1);                            %The Covariance Matrix (M*N)*(M*N) (Need Time)

[EVectors, EValues] = eig(Q);

EVals = diag(EValues);

[EValSorted, Index] = sort(EVals, 'descend');                          % Sort the eigen values descending (the largest first)
EVecSorted = EVectors(:, Index);

%% Principal Components
PCA_Vectors = EVecSorted(:, 1:L);                                      %Largest L eigen vectors correspond to largest L eigen values

%% Projecting the copy of the resized images into the L Principal Components
for i=1:n
    PCA_Images(i, :) = (resized_images_copy(i, :)-images_mean) * PCA_Vectors;
end
