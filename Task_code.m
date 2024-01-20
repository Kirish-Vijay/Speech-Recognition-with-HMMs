filepath = "~/Downloads/EEEM030cw2_DevelopmentSet/";
%change filepath as needed
soundfiles = dir(fullfile(filepath, '*.mp3'));
%gets all the files ending in .mp3 in the directory
featurearray = cell(1,length(soundfiles));
filecount = length(soundfiles);
%init map for ground truth labels
groundTruthMap = containers.Map();
for i = 1:length(soundfiles)
   filename = soundfiles(i).name;
   [~, name, ~] = fileparts(filename);
   parts = strsplit(name, '_');
   word = parts(end); %get last part of filename (the word)
   groundTruthMap(filename) = word;
end
%array to store mfcc data
for i = 1:length(soundfiles)
   [sound,fs] = audioread(fullfile(filepath,soundfiles(i).name));
   %reads the audio file
   melceps = mfcc(sound,fs, 'Window',hamming(round(0.03*fs), 'periodic'),'OverlapLength',round(0.02*fs), 'NumCoeffs', 12);
   %gets the mel cepstrum coefficients
   featurearray{i} = melceps;
   % stores in the array
end
mfccmatrix = vertcat(featurearray{:});
%converts to a matrix
meanvals = mean(mfccmatrix);
%gets the mean values of the mfcc matrix
covariance = cov(mfccmatrix);
for i= 1:13
   for j = 1:13
       if i~= j
           covariance(i,j) = 0;
       end
   end
end
%gets the covariance of the mfcc matrix
avg_duration = round(length(mfccmatrix) / filecount);
%mfcc matrix is the same length as the length of all frames per state
%we divide by the file count to get the average duration
aii = exp(-1/(avg_duration-1));
N = 8;
K = 13;
%parameters given in the brief
A = zeros(N,N);
for i= 1:N
   for j = 1:N
       if i== j
           A(i,j) = aii;
       else if i == (j-1)
               A(i,j) = (1-aii);
       end
       end
   end
end
entryA = [1,0,0,0,0,0,0,0];
exitA = [0,0,0,0,0,0,0,(1-aii)];
%initialises the state transition probability with a matrix with each row
%summing to one
%since it is flat start, these are generic.
pi = ones(1, N)/ N;
%initialises the initial state probability with a matrix thats
%probabilities add to one
%since it is flat start, these are a generic value.
B.mu = repmat(meanvals,N,1);
B.sig = repmat(covariance,[1, 1, N]);
%B is the output probabilities. Mu is the mean and sig is the covariance
disp('Transition Probabilities (A):');
disp(A);
disp('Initial State Probabilities (pi):');
disp(pi);
disp('Output Probabilities (B):');
disp(B);
obsfeat = featurearray{29};
obf = obsfeat(3,:);
bprob = zeros(K, N);
for state = 1:N
   meanvec = B.mu(state, :);
   covvec = B.sig(:, :, state);
   % Calculate probabilities for all frames at once
   bprob(:, state) = mgpdf(obf, meanvec, covvec);
end
disp("done");
%end of task 2
% Task 3: Baum-Welch Training for HMM (well crafted code for task 3)
iteration_max = 15; % Number of iterations for Baum-Welch training
for iteration = 1:iteration_max
   % Forward procedure
   [alpha, forwardlikelihoods] = forwardproc(pi, bprob, A, aii);
   disp(forwardlikelihoods);
   % Backward procedure
   beta = backwardsproc(A, bprob, aii);
   disp(beta);
   % Compute occupation likelihoods
   gamma = occupation(alpha, beta, forwardlikelihoods);
   % Update state transition probabilities
   newA = transitionlikelihoods(gamma, A, bprob);
   A = newA;
end
% Apply Viterbi algorithm
optimalPath = viterbi(bprob, A, pi);
% Display the optimal state sequence
disp('Optimal State Sequence:');
disp(optimalPath);
% Task 4b: Evaluate the recognizer after each iteration
% Evaluation of recognizer
errorRate = 0;
for i = 1:length(featurearray)
   bprob = calculateBprob(featurearray{i}, B);
   optimalPath = viterbi(bprob, A, pi);
   recognizedWord = mapStateToWord(optimalPath, groundTruthMap, soundfiles(i).name);
   actualWord = groundTruthMap(soundfiles(i).name);
   if ~strcmp(recognizedWord, actualWord)
       errorRate = errorRate + 1;
   end
end
errorRate = errorRate / length(featurearray);
fprintf('Iteration %d: Error Rate = %.2f%%\n', iteration, errorRate*100);


%Task 5
testfilepath = "~/Downloads/EEEM030cw2_EvaluationSet/";
%change filepath as needed
audiofiles = dir(fullfile(filepath, '*.mp3'));
%gets all the files ending in .mp3 in the directory
testfeaturearray = cell(1,length(soundfiles));
testfilecount = length(soundfiles);
confusionMatrixWords = ["heed","hid","head","had","hard","hud","hod","hoard","hood","whod","heard","again","say"];
confusionMatrixIndex = 1:length(confusionMatrixWords);
confusionDictionary = dictionary(confusionMatrixWords,confusionMatrixIndex);
confusionMatrix = zeros(length(confusionMatrixWords)); %Number of words we trained with
for i = 1:length(audiofiles)
   [audio,fs] = audioread(fullfile(filepath,audiofiles(i).name));
   %reads the audio file
   melceps = mfcc(audio,fs, 'Window',hamming(round(0.03*fs), 'periodic'),'OverlapLength',round(0.02*fs), 'NumCoeffs', 12);
   %gets the mel cepstrum coefficients
   testfeaturearray{i} = melceps;
   % stores in the array
   errorRate = 0;
   bprob = calculateBprob(melceps, B);
   optimalPath = viterbi(bprob, A, pi);
   recognizedWord = mapStateToWord(optimalPath, groundTruthMap, soundfiles(i).name);
   actualWord = groundTruthMap(soundfiles(i).name);
   %Generates the recognized and actual word then displays them
   disp("Recognized word:");
   disp(recognizedWord);
   disp("Actual word;");
   disp(actualWord);
   if ~strcmp(recognizedWord, actualWord)
       errorRate = errorRate + 1;
   End


%Updates the confusion matrix
   recognizedIndex = confusionDictionary(recognizedWord{1});
   actualIndex = confusionDictionary(actualWord{1});
   confusionMatrix(recognizedIndex,actualIndex) = confusionMatrix(recognizedIndex,actualIndex)+1;
end
for index=1:length(confusionMatrixWords)
   confusionMatrix(index,:) = confusionMatrix(index,:)/sum(confusionMatrix(index,:));
End


%Display the confusion matrix along with the error rate
disp("Confusion Matrix");
disp(confusionMatrix);
errorRate = errorRate / length(testfeaturearray);
fprintf('Error Rate = %.2f%%\n', errorRate*100);


function probability = mgpdf(X, mu, sig)
d = size(mu, 2);
numFrames = size(X, 1);
% Replicate mean and cov for each frame
mu_rep = repmat(mu, numFrames, 1);
sig_rep = repmat(sig, [1, 1, numFrames]);
% Extract the diagonal elements of the covariance matrix for each frame
diag_sig = zeros(numFrames, size(sig, 1));
for frame = 1:numFrames
   diag_sig(frame, :) = diag(squeeze(sig_rep(:, :, frame)));
end
% Exponent term
exponent = -0.5 * sum((X - mu_rep) .* (1./diag_sig) .* (X - mu_rep), 2);
% Denominator term
denom = (2 * pi)^(d/2) * sqrt(prod(diag_sig, 2));
% Probability for each frame
probability = 1 ./ denom .* exp(exponent);
end
function [ALPH,total_prob_observe] = forwardproc(pi,B1,A,aii)
ALPH= zeros (13,8);
sum1 = 0;
for i = 1:8
   ALPH(1,i) = pi(i)* B1(i,1);
end
for t = 2:13
   for j = 1:8
       for i = 1:8
           sum1 = sum1 + ALPH(t - 1, i) * A(i, j);
       end
       ALPH(t, j) = sum1 * B1(t,j);
       sum1 = 0;
   end
end
total_prob_observe = 0;
for i = 1:8
   total_prob_observe = total_prob_observe + ALPH(13, i) * (aii); %exit_prob, need to identified in matrix A
end
end
function BETA = backwardsproc (A,B1,aii)
sum1 = 0;
T = 13;
N = 8; %no. of states
BETA(T, N) = 0;
for i = 1:N
   BETA(T, i) = aii; %exit probility
end
%for state 1-12
for t = T-1:-1:1
   for i = 1:N
       for j = 1:N
           sum1 = sum1 + A(i, j) * B1(t+1, j) * BETA(t+1, j);
       end
       BETA(t, i) = sum1;
       sum1 = 0;
   end
end
end
function GAMMA = occupation(ALPH, BETA,norm)
T = size(ALPH, 1); % Number of time frames
N = 8; % Number of states
GAMMA = zeros(T, N); % Initialize occupation likelihoods
for t = 1:T
   % Calculate occupation likelihood for each state
   GAMMA(t, :) = (ALPH(t, :) .* BETA(t, :)) / norm;
end
end
function newA = transitionlikelihoods(gamma, A, B1)
[T, N] = size(gamma);
numerator = zeros(N, N);
denominator = zeros(N, 1);
for t = 1:T-1
   for i = 1:N
       denominator(i) = denominator(i) + gamma(t, i);
       for j = 1:N
           numerator(i, j) = numerator(i, j) + gamma(t, i) * A(i, j) * B1(t, j);
       end
   end
end
newA = numerator ./ denominator;
end
function recognizedWord = mapStateToWord(optimalPath, groundTruthMap, filename)
% Use the filename to get the ground truth word
if isKey(groundTruthMap, filename)
   recognizedWord = groundTruthMap(filename);
else
   recognizedWord = 'Unknown';
end
end
function bprob = calculateBprob(features, B)
% features: A matrix of features for a single audio file
% B: The B structure containing mu and sig for each state
K = size(features, 1); % Assuming features are rows of MFCCs
N = size(B.mu, 1); % Number of states
bprob = zeros(K, N);
for state = 1:N
   meanvec = B.mu(state, :);
   covvec = B.sig(:, :, state);
   % Calculate probabilities for all frames at once
   bprob(:, state) = mgpdf(features, meanvec, covvec);
end
end
% Task 4: Viterbi Algorithm for Optimal State Sequence
function path = viterbi(B, A, pi)
T = size(B, 1); % Total number of observations
N = size(A, 1); % Number of states
delta = zeros(T, N);
psi = zeros(T, N);
% Initialization
delta(1, :) = pi .* B(1, :);
psi(1, :) = 0;
% Recursion
for t = 2:T
   for j = 1:N
       [max_val, max_index] = max(delta(t-1, :) .* A(:, j)');
       delta(t, j) = max_val * B(t, j);
       psi(t, j) = max_index;
   end
end
% Termination
[P, last_state] = max(delta(T, :));
path = zeros(1, T);
path(T) = last_state;
% Path backtracking
for t = T-1:-1:1
   path(t) = psi(t + 1, path(t + 1));
end
end