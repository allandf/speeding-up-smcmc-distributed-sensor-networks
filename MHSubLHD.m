function [phi,Lamda,ProxM,t,gamma,delta,sample_std,cc,C] = MHSubLHD(z,x,xP,H,R,delta,T)
% MH with adaptive subsampling for AS-SMCMC.
%
% Inputs:
%   z       : full measurement set 
%   x,xP    : current and proposed states
%   H,R     : measurement model and covariance
%   delta   : confidence parameter for stopping rule
%   T       : struct with Taylor proxy data
%
% Outputs:
%   phi         : normalized log-threshold
%   Lamda       : subsampled log-lik ratio estimate 
%   ProxM       : average proxy correction
%   t           : subsample size at stopping
%   gamma       : growth factor for subsample size
%   delta       : updated confidence parameter
%   sample_std  : running std of corrected log-lik ratios
%   cc          : [|est+ProxM-phi|, bound] history for diagnosing stop
%   C           : constant used in Bernstein bound

n = size(z,1); 

p = 1.1; % user defined input 
phi = (1/n) * log(rand);

gamma = 1.2; % geometric growth factor for subsample size

t = 0;       % number of samples processed
tlook = 1;   % iteration counter
Lamda_est = 0; % running mean (estimate) of log-lik ratio
b = 1;        % current subsample size
DONE = 1;     % stopping flag

C=0.5*(abs((H*x')'-T.point)*T.MatTmax*abs((H*x')'-T.point)'+abs((H*xP')'-T.point)*T.MatTmax*abs((H*xP')'-T.point)');

Prox = sum(T.T1.*repmat(((H*xP')'-(H*x')'),length(T.T1),1),2);

ProxM = mean(Prox);

ind = randperm(size(z,1)); % sample without replacement
SavedLogLikRatio=[];
  
while(DONE) 
    % Likelihoods on the next block of measurements
    likP = ComputationLikelihoodG(z(ind(t+1:b),:),(H*xP')',R); % proposed
    lik  = ComputationLikelihoodG(z(ind(t+1:b),:),(H*x')',R);  % current
    
    SavedLogLikRatio=[SavedLogLikRatio (log(likP./lik)-Prox(ind(t+1:b)))'];

    sample_std(tlook)=std(SavedLogLikRatio);
    
    Lamda_est = (1/b) * (t*Lamda_est + sum(log(likP./lik) - Prox(ind(t+1:b))));
    
    t = b;
    deltatlook = (((p-1)/(p*tlook^p))*delta);
    
    c = sample_std(tlook)*sqrt((2*log(3/deltatlook))/(t)) + 6*C*log(3/deltatlook)/t;

    tlook = tlook + 1;
    b =  min(n,ceil(gamma*t)); % grow subsample geometrically, cap at n
    cc(tlook,:)=[abs(Lamda_est + ProxM - phi),c];

    % Stopping rule: when estimate is confidently on one side of phi,
    % or we've exhausted the dataset (t==n).
    if (abs(Lamda_est + ProxM - phi) > c) || (t == n)
        DONE = 0;
    end
end

Lamda = Lamda_est; 
end
