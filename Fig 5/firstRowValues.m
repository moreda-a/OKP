%% generate values
clear all

%% generate arrival instance
load('dataIns')
K = length(dataIns); % no. of instances
theta = 250;

for k = 1:K
    instanceName = ['arrival_Instance/arrIns', num2str(k)];
    load(instanceName);

    N = size(arrIns, 1);

    %% generate value and weight for each job
    randv = unifrnd(1, theta, N, 1); % uniformly distributed in [1, \theta] for values
    weightCandidate = [0.01, 0.03, 0.05];
    randw = floor(unifrnd(1, 4, N, 1));
    jobWeight = weightCandidate(randw)'; % Job weights
    jobValue = randv .* arrIns(:, 3) .* jobWeight; % Job values

    %% save value and weight as two columns
    resultMatrix = [jobValue, jobWeight]; % Two columns matrix
    valueName = ['values250/jobvalue', num2str(k), '.csv']; % Saving as CSV for easier inspection
    writematrix(resultMatrix, valueName); % Save as CSV with jobValue and jobWeight
end