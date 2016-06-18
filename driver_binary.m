load Data\mnist_uint8.mat
train_x=double(train_x);
train_y=double(train_y);
test_x=double(test_x);
test_y=double(test_y);

X=double(train_x);
X = X';
[train_evecs,train_evals,mean,train_meandata]=compute_pca(X);

k = 0;
denominator = sum(train_evals);
variance_retained = sum(train_evals(1:k))/denominator ;
while (k <= size(train_evals,1)-1)&&(variance_retained <=.70)
    k=k+1;
    variance_retained = sum(train_evals(1:k))/denominator;
end

size_x = size(test_x,1);
for i = 1:size_x
 test_x(i,:) = test_x(i,:)-mean;
end

train_transformationMat = train_evecs(:,1:k) ;
train_reducedData = train_meandata * train_transformationMat;

test_transformationMat = train_evecs(:,1:k) ;
test_reducedData = test_x * test_transformationMat;

label_test = zeros(size(test_reducedData,1),size(train_y,2));
for i = 1:size(test_reducedData,1)
    diff = repmat(test_reducedData(i,:),size(train_x,1),1) - train_reducedData;
    distance = diff.^2;
    distance_matrix = sum(distance,2);
    [mindistance,minIndex] = min(distance_matrix);
    label_test(i,:) = train_y(minIndex,:);
end

count = 0;
for i =1:size(test_x,1)
    testpoint = test_y(i,:);
    [~,tclass] = max(testpoint);
    [~,classfied] = max(label_test(i,:));
    if(tclass == classfied)
        count = count+1;
    end
end

accuracy = (count/size(test_x,1))*100;
accuracy
