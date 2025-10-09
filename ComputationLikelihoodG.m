function [Likelihood]=ComputationLikelihoodG(z,mean,cov)
for jj= 1:length(z)
    Likelihood(jj,1)=mvnpdf(z(jj),mean,cov);
end
