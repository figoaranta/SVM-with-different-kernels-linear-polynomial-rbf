close all;
clear;

% <-- Load Processed Dataset for both regression and classification -->
load("pearson_air_quality.mat");
% load("final_name_gender_dataset.mat");

% final_x_train = final_x_train(1:10000,:);
% final_y_train = final_y_train(1:10000,:);

% <-- Task 1 Regression SVM -->
c_svm = fitcsvm(final_x_train,final_y_train, 'KernelFunction','linear', 'BoxConstraint',2);


gender_dataset = [X,y];
cv = cvpartition(size(gender_dataset,1),'HoldOut',0.3);
idx = cv.test;
dataTrain = gender_dataset(~idx,:);
dataTest  = gender_dataset(idx,:);

x_train = dataTrain(:,(1:2));
y_train = dataTrain(:,(3));

x_test = dataTest(:,(1:2));
y_test = dataTest(:,(3));

rmse_score = [];
epsi = [1,5,10,15];

for i=1:length(epsi)
    r_svm = fitrsvm(x_train,y_train,'KernelFunction','rbf','BoxConstraint',1,'Epsilon',epsi(i));
    y_hat = predict(r_svm,x_test);
    rmse = sqrt(mean((y_test - y_hat).^2));
    rmse_score = [rmse_score,rmse];
end
disp(rmse_score);


% disp(rmse_score);
% <-- Task 2 Nested Cross Validation -->
clear;
hyperparams = containers.Map();
outerLoop = 10;
innerLoop = 2;

r_nestedCrossValidation = r_nested_cross_validation(X,y,outerLoop,innerLoop);
c_nestedCrossValidation = c_nested_cross_validation(final_x_train,final_y_train,outerLoop,innerLoop);

hyperparams('r_best_c_rbf') = r_nestedCrossValidation("c_rbf");
hyperparams('r_best_epsilon_rbf') = r_nestedCrossValidation("epsilon_rbf");
hyperparams('r_best_sigma_rbf') = r_nestedCrossValidation("sigma_rbf");

hyperparams('r_best_c_poly') = r_nestedCrossValidation("c_poly");
hyperparams('r_best_epsilon_poly') = r_nestedCrossValidation("epsilon_poly");
hyperparams('r_best_q_poly') = r_nestedCrossValidation("q_poly");

hyperparams('c_best_c_rbf') = c_nestedCrossValidation("c_rbf");
hyperparams('c_best_sigma_rbf') = c_nestedCrossValidation("sigma_rbf");

hyperparams('c_best_c_poly') = c_nestedCrossValidation("c_poly");
hyperparams('c_best_q_poly') = c_nestedCrossValidation("q_poly");





% <-- Task 3 10-fold Cross-validation using tuned hyperparameters from task 2 -->
% Regression Cross-Validation
rmse_score_rbf = [];
rmse_score_poly = [];
k = 10;
length_of_fold = floor(length(X)/k);
for i=1:k
    x_test = X((k-i)*length_of_fold+1 : (k-i+1)*length_of_fold , :);
    y_test = y((k-i)*length_of_fold+1 : (k-i+1)*length_of_fold);

    x_train = X;
    x_train((k-i)*length_of_fold+1 : (k-i+1)*length_of_fold , :) = [];
    y_train = y;
    y_train((k-i)*length_of_fold+1 : (k-i+1)*length_of_fold) = [];


    r_svm_rbf = fitrsvm(X,y,"KernelScale",hyperparams('r_best_sigma_rbf'),"Epsilon",hyperparams('r_best_epsilon_rbf'),"BoxConstraint",hyperparams('r_best_c_rbf'));
    r_svm_poly = fitrsvm(X,y,"KernelFunction","polynomial","PolynomialOrder",hyperparams('r_best_q_poly'),"Epsilon",hyperparams('r_best_epsilon_poly'),"BoxConstraint",hyperparams('r_best_c_poly'));
    
    yhat_rbf = predict(r_svm_rbf,x_test);
    yhat_poly = predict(r_svm_poly,x_test);
    
    rmse_rbf = sqrt(mean((y_test - yhat_rbf).^2));
    rmse_poly = sqrt(mean((y_test - yhat_poly).^2));
    
    rmse_score_rbf = [rmse_score_rbf,rmse_rbf ];
    rmse_score_poly = [rmse_score_poly,rmse_poly ];
end

avg_rmse_rbf = sum(rmse_score_rbf)/length(rmse_score_rbf);
avg_rmse_poly = sum(rmse_score_poly)/length(rmse_score_poly);

disp("---------------KFold_Cross_Validation Regression---------------")
disp("RMSE for RBF Regression SVM: "+avg_rmse_rbf)
disp("RMSE for Poly Regression SVM: "+avg_rmse_poly)


% Classification Cross-Validation
accuracy_kfold_rbf = [];
accuracy_kfold_poly = [];

length_of_fold_c = floor(length(final_x_train)/k);
for i = 1:outerLoop
    x_test = final_x_train((outerLoop-i)*length_of_fold_c+1 : (outerLoop-i+1)*length_of_fold_c , :);
    y_test = final_y_train((outerLoop-i)*length_of_fold_c+1 : (outerLoop-i+1)*length_of_fold_c);

    x_train = final_x_train;
    x_train((outerLoop-i)*length_of_fold_c+1 : (outerLoop-i+1)*length_of_fold_c , :) = [];
    y_train = final_y_train;
    y_train((outerLoop-i)*length_of_fold_c+1 : (outerLoop-i+1)*length_of_fold_c) = [];

    svm_c_rbf_kfold_optimized = fitcsvm(x_train,y_train,"KernelFunction","rbf","KernelScale",hyperparams('c_best_sigma_rbf'),"BoxConstraint",hyperparams('c_best_c_rbf'));
    svm_c_poly_kfold_optimized = fitcsvm(x_train,y_train,"KernelFunction","polynomial","PolynomialOrder",hyperparams('c_best_q_poly'),"BoxConstraint",hyperparams('c_best_c_poly'));

    y_pred_rbf = predict(svm_c_rbf_kfold_optimized,x_test);
    y_pred_poly = predict(svm_c_poly_kfold_optimized,x_test);

    accuracy_kfold_rbf = [accuracy_kfold_rbf , (sum(y_pred_rbf==1 & y_test==1) + sum(y_pred_rbf==0 & y_test==0))/length(y_test)*100 ];
    accuracy_kfold_poly = [accuracy_kfold_poly , (sum(y_pred_poly==1 & y_test==1) + sum(y_pred_poly==0 & y_test==0))/length(y_test)*100 ];
end


disp("---------------KFold_Cross_Validation Classification---------------")
disp("RBF Accuracy Score: "+(sum(accuracy_kfold_rbf)/length(accuracy_kfold_rbf))/100);
disp("Polynomial Accuracy Score: "+(sum(accuracy_kfold_poly)/length(accuracy_kfold_poly))/100);
