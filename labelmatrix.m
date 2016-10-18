% Code based on
% http://stackoverflow.com/questions/25109239/generate-matrix-of-vectors-from-labels-for-multiclass-classification-vectorized
function Y = labelmatrix(y, k)
  m = length(y);
  Y = repmat(y(:),1,k) == repmat(0:k-1,m,1);
end