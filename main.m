%% Phase Retrieval Demo - SCAD Regularized Phase Retrieval
%
% This demo demonstrates single-shot phase retrieval from intensity
% measurements using SCAD (Smoothly Clipped Absolute Deviation) regularization
% with gradient-sparse non-convex regularization integrating physical constraints.
%
% Reference:
% Chen, X., Li, F. Single-Shot Phase Retrieval Via Gradient-Sparse
% Non-Convex Regularization Integrating Physical Constraints.
% J Sci Comput 102, 63 (2025).
% https://doi.org/10.1007/s10915-025-02788-2

clear all;
close all;
clc;

addpath("./func");

%% Simulation Parameters
params.pxsize = 5e-3; % pixel size (mm)
params.wavlen = 0.5e-3; % wavelength (mm)
params.method = 'Angular Spectrum'; % propagation method: 'Fresnel' or 'Angular Spectrum'
params.dist = 5; % propagation distance (mm)

%% Algorithm Parameters
mu = 0.05; % penalty parameter
alpha = 2; % step size
gamma = 3.7; % SCAD parameter
lambda = 0.1; % regularization parameter
eta = 0.15; % SCAD threshold

Tm = 2000; % iterations
Tsub = 3; % sub-iterations

%% Load and Prepare Test Image
n = 256;
img = imresize(im2double(imread('imgs/house.png')), [n, n]);

x0 = exp(1i * pi * img); % phase-only

kernelsize = params.dist * params.wavlen / params.pxsize / 2;
nullpixels = ceil(kernelsize / params.pxsize);
x = zeropad(x0, nullpixels);

fprintf('Image size: %dx%d \n', n, n);

%% Forward Model Setup
Q = @(x) propagate(x, params.dist, params.pxsize, params.wavlen, params.method);
QH = @(x) propagate(x, -params.dist, params.pxsize, params.wavlen, params.method);

C = @(x) imgcrop(x, nullpixels);
CT = @(x) zeropad(x, nullpixels);

A = @(x) C(Q(x));
AH = @(x) QH(CT(x));

%% Generate Intensity Measurements
y = abs(A(x)) .^ 2;

% Add noise to the measurements (optional)
% rng(0)
% y0 = abs(A(x)) .^ 2;
% y = awgn(y0, SNR, 'measured', 0); % SNR (dB)
% y(y < 0) = 0;

fprintf('Intensity measurements generated.\n');

%% Phase Retrieval Setup
region.x1 = nullpixels + 1;
region.x2 = nullpixels + n;
region.y1 = nullpixels + 1;
region.y2 = nullpixels + n;

support = zeros(size(x));
support(nullpixels + 1:nullpixels + n, nullpixels + 1:nullpixels + n) = 1;
absorption = 1; % absorption upper bound

projf = @(x) min(abs(x), absorption) .* exp(1i * angle(x)) .* support;

x_init = AH(sqrt(y));

%% GPU Setup
fprintf('Setting up GPU computation...\n');
device = gpuDevice();
x = gpuArray(x);
x0 = gpuArray(x0);
x_init = gpuArray(x_init);
y = gpuArray(y);
support = gpuArray(support);

% Evaluation functions
RMSEFunc = @(z) phase_RMSE(z, x0);
CWSSIMFunc = @(z) cwssim_index(z, x0, 6, 16, 0, 0);

%% ADMM Optimization
fprintf('Starting phase retrieval optimization on GPU...\n');

x_iter = x_init;
z = D(x_iter);
w = 0 * z;
x_prev = 0 * x_init;

for k = 1:Tm

    for k_sub = 1:Tsub

        grad = 1/2 * AH(exp(1i * angle(A(x_iter))) .* (abs(A(x_iter)) - sqrt(y))) + ...
            (mu / 2) * DH(D(x_iter) + w - z);
        x_iter = x_iter - alpha * grad;

        x_iter = projf(x_iter);

        x_iter = x_iter + k_sub / (k_sub + 3) * (x_iter - x_prev);
        x_prev = x_iter;
    end

    z = Shrink_SCAD(D(x_iter) + w, gamma, lambda, eta / mu);

    w = w + D(x_iter) - z;

    % metric evaluation every 100 iters
    if mod(k, 100) == 0
        Rx0 = x_iter(region.x1:region.x2, region.y1:region.y2);
        Rx = RPhase(Rx0, x0);
        current_RMSE = RMSEFunc(Rx);
        current_CWSSIM = CWSSIMFunc(Rx);
        fprintf('Iter: %4d, RMSE: %.4f, CW-SSIM: %.4f \n', k, current_RMSE, current_CWSSIM);
    end

    clear mex;
end

% final reconstructed
Rx0 = x_iter(region.x1:region.x2, region.y1:region.y2);
Rx = RPhase(Rx0, x0);
x_final = Rx;
final_RMSE = RMSEFunc(x_final);
final_CWSSIM = CWSSIMFunc(x_final);

wait(device);

% Gather results from GPU
x0 = gather(x0);
x_final = gather(x_final);
final_RMSE = gather(final_RMSE);
final_CWSSIM = gather(final_CWSSIM);
y = gather(y);

fprintf('Optimization completed.\n');

%% Display Results
figure('Name', 'Phase Retrieval Results', 'Position', [100, 100, 900, 400]);

% Original amplitude and phase
subplot(2, 3, 1);
imshow(abs(x0), [0, 1]);
title('GT Amplitude');
colorbar;

subplot(2, 3, 4);
imshow(angle(x0), [0, pi]);
title('GT Phase');
colorbar;

% Simulated intensity measurement
subplot(2, 3, 2);
imshow(y, []);
title('Intensity Measurement');
colorbar;

% Reconstructed amplitude and phase
subplot(2, 3, 3);
imshow(abs(x_final), [0, 1]);
title('Reconstructed Amplitude');
colorbar;

subplot(2, 3, 6);
imshow(angle(x_final), [0, pi]);
title('Reconstructed Phase');
colorbar;

% Print final metrics
fprintf('\n=== Final Results ===\n');
fprintf('Final RMSE: %.4f\n', final_RMSE);
fprintf('Final CW-SSIM: %.4f\n', final_CWSSIM);

%% Helper Functions

function u = imgcrop(x, cropsize)
    u = x(cropsize + 1:end - cropsize, cropsize + 1:end - cropsize);
end

function u = zeropad(x, padsize)
    u = padarray(x, [padsize, padsize], 0);
end

function d = D(x)
    % Discrete gradient operator
    [n1, n2] = size(x);
    w = zeros(n1, n2, 2);

    w(:, :, 1) = x - circshift(x, [-1, 0]);
    w(n1, :, 1) = 0;

    w(:, :, 2) = x - circshift(x, [0, -1]);
    w(:, n2, 2) = 0;

    d = w;
end

function dh = DH(w)
    % Adjoint of discrete gradient operator
    [n1, n2, ~] = size(w);

    shift = circshift(w(:, :, 1), [1, 0]);
    u1 = w(:, :, 1) - shift;
    u1(1, :) = w(1, :, 1);
    u1(n1, :) = -shift(n1, :);

    shift = circshift(w(:, :, 2), [0, 1]);
    u2 = w(:, :, 2) - shift;
    u2(:, 1) = w(:, 1, 2);
    u2(:, n2) = -shift(:, n2);

    dh = u1 + u2;
end
