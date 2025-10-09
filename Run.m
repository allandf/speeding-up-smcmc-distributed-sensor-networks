clc;clear;close all;
% Driver script: simulate data, partition into shards, and run KF + SMCMC variants.

load Gtraj.mat            % load true trajectory and total simulation time
P.N = 5000;               % Number of particles/MCMC steps
P.NBurn = round(P.N/4);   % Burn in period

P.S = 4; P.K = 2; P.d = dim;P.A = A;
P.NC = P.S*P.N;P.Q = Q*eye(dim);
P.Tot_t = Tot_t;P.x_actual = x_actual;

P.Nmeas = 500;      % Number of measurements

% measurement model
P.H = ones(1,dim); P.R = 2*eye(dim);

% measurement and state generation
for t = 2:P.Tot_t
    % Simulate measurements: Nmeas draws around H*x_actual(t,:)
    P.Z{t} = mvnrnd(repmat((P.H*x_actual(t,:)')',P.Nmeas,1),P.R);
    % Vectorized stack for Kalman filter convenience
    P.zz(:,t) = reshape(P.Z{t}',1,dim*P.Nmeas)';
    % Randomly partition measurements into S shards (roughly equal sized)
    ind1 = randperm(P.Nmeas);
    for s = 1:P.S-1
        P.z{t,s} = P.Z{t}(ind1(floor(P.Nmeas/P.S)*(s-1)+1:floor(P.Nmeas/P.S)*(s)),:);
    end
    P.z{t,P.S} = P.Z{t}(ind1(floor(P.Nmeas/P.S)*(P.S-1)+1:end),:);
end

% Algorithm computation: KF and three SMCMC variants
[Kalman] = KF(P);
[SMCMC_G, A1_SMCMC, A2_SMCMC] = SMCMC(P);
[SMCMC_AS, A1_SMCMC_AS, A2_SMCMC_AS,NUM_MU] = SMCMC_AS(P);
[SMCMC_EP, A1_DSMCMC_EP, A2_DSMCMC_EP] = SMCMC_EP(P);
