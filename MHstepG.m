function [phi,Lamda] = MHstepG(z,x,xP,H,R)
% Metropolis–Hastings log-acceptance ratio for generic SMCMC.
% Uses full measurement set z at current time.

phi = (1/size(z,1)) * (log(rand)); % normalized log-threshold

% Per-measurement likelihoods under proposed/current
for jj = 1:length(z)
    likP(jj) = mvnpdf(z(jj),(H*xP).',R);
    lik(jj) = mvnpdf(z(jj),(H*x).',R);
end

% Mean log-likelihood ratio
Lamda = (1/size(z,1)) * sum(log(likP./lik));

end
