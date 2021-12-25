function bestParams = c_nested_cross_validation(X,y,outerFold,innerFold)

    
    hyperparam = containers.Map();
    
    hyperparam("constraint")= {1,1,1};
    hyperparam("sigma")= {5,10,20};
    hyperparam("q")= {1,2,3};
    
    outerLoop = outerFold;
    innerLoop = innerFold;
    
    length_of_fold = floor(length(X)/outerLoop);
    
    hyperparam_vector_rbf = [];
    accuracy_score_rbf = [];
    
    hyperparam_vector_poly =[];
    accuracy_score_poly = [];
    
    for i=1:outerLoop
        x_test = X((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold , :);
        y_test = y((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold);
    
        x_train = X;
        x_train((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold , :) = [];
        y_train = y;
        y_train((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold) = [];
    
        length_of_inner_fold = floor(length(x_train)/innerLoop);
    
        constraint = cell2mat(hyperparam("constraint"));
        sigma = cell2mat(hyperparam("sigma"));
        q = cell2mat(hyperparam("q"));

        accuracy_training_rbf = zeros(1,length(constraint)*length(sigma));
        accuracy_training_poly = zeros(1,length(constraint)*length(q));
    
        for j=1:innerLoop
            x_validation_set = x_train((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: );
            y_validation_set = y_train((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: );
        
            x_training_subset = x_train;
            x_training_subset((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: ) = [];
            y_training_subset = y_train;
            y_training_subset((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: ) = [];
        
    
            for k = 1:length(constraint)
                constraint_inner = constraint(k);
                
                for l = 1:length(sigma)
                    sigma_inner = sigma(l);
        
                    svm_c_rbf = fitcsvm(x_training_subset,y_training_subset,"KernelFunction","rbf","KernelScale",sigma_inner,"BoxConstraint",constraint_inner);
                    y_pred_rbf = predict(svm_c_rbf,x_validation_set);
        
                    accuracy_inner_rbf = (sum(y_pred_rbf==1 & y_validation_set==1) + sum(y_pred_rbf==0 & y_validation_set==0))/length(y_validation_set)*100;
        
                    accuracy_training_rbf(l + ((k-1)* length(sigma)) ) = accuracy_training_rbf(l + (k-1)* l ) + accuracy_inner_rbf;
                end
        
        
                for l = 1:length(q)
                    q_inner = q(l);
                    
                    svm_c_poly = fitcsvm(x_training_subset,y_training_subset,"KernelFunction","polynomial","PolynomialOrder",q_inner,"BoxConstraint",constraint_inner);
                    y_pred_poly = predict(svm_c_poly,x_validation_set);
                
                    accuracy_inner_poly = (sum(y_pred_poly==1 & y_validation_set==1) + sum(y_pred_poly==0 & y_validation_set==0))/length(y_validation_set)*100;
        
                    accuracy_training_poly(l + ((k-1)* length(q)) ) = accuracy_training_poly(l + (k-1)* l ) + accuracy_inner_poly;
        
                end
            end
        end
    
        [best_hyperparam_score_rbf,best_hyperparam_rbf] = min(accuracy_training_rbf);
        best_hyperparam_rbf = best_hyperparam_rbf - 1 ;
        hyperparam_vector_rbf = [hyperparam_vector_rbf,best_hyperparam_rbf];
    
        best_hyperparam_rbf_constraint = floor((best_hyperparam_rbf)/length(sigma))+1;
        best_hyperparam_rbf_sigma = mod(best_hyperparam_rbf,length(sigma))+1;
    
        [best_hyperparam_score_poly,best_hyperparam_poly] = min(accuracy_training_poly);
        best_hyperparam_poly = best_hyperparam_poly - 1 ;
        hyperparam_vector_poly = [hyperparam_vector_poly,best_hyperparam_poly];
    
        best_hyperparam_poly_constraint = floor((best_hyperparam_poly)/length(q))+1;
        best_hyperparam_poly_q = mod(best_hyperparam_poly,length(q))+1;
    
        constraint_outer_rbf = constraint(best_hyperparam_rbf_constraint);
        sigma_outer_rbf = sigma(best_hyperparam_rbf_sigma);
    
        constraint_outer_poly = constraint(best_hyperparam_poly_constraint);
        q_outer_poly = q(best_hyperparam_poly_q);
    
        svm_c_rbf = fitcsvm(x_train,y_train,"KernelFunction","rbf","KernelScale",sigma_outer_rbf,"BoxConstraint",constraint_outer_rbf);
        y_pred_rbf = predict(svm_c_rbf,x_test);

        accuracy_outer_rbf = (sum(y_pred_rbf==1 & y_test==1) + sum(y_pred_rbf==0 & y_test==0))/length(y_test)*100;
        accuracy_score_rbf = [accuracy_score_rbf, accuracy_outer_rbf];
    
        svm_c_poly = fitcsvm(x_train,y_train,"KernelFunction","polynomial","PolynomialOrder",q_outer_poly,"BoxConstraint",constraint_outer_poly);
        y_pred_poly = predict(svm_c_poly,x_test);
    
        accuracy_outer_poly = (sum(y_pred_poly==1 & y_test==1) + sum(y_pred_poly==0 & y_test==0))/length(y_test)*100;
        accuracy_score_poly = [accuracy_score_poly, accuracy_outer_poly ];
    
    end
    
    hyperparam_constraint = cell2mat(hyperparam("constraint"));
    hyperparam_sigma = cell2mat(hyperparam("sigma"));
    hyperparam_q = cell2mat(hyperparam("q"));
    
    best_hyperparam_rbf_constraint = floor((mode(hyperparam_vector_rbf))/length(sigma))+1;
    best_hyperparam_rbf_sigma = mod(mode(hyperparam_vector_rbf),length(sigma))+1;
    
    svm_c_rbf_optimized = fitcsvm(X,y,"KernelFunction","rbf","KernelScale",hyperparam_sigma(best_hyperparam_rbf_sigma),"BoxConstraint",hyperparam_constraint(best_hyperparam_rbf_constraint));
    support_vector_rbf = histc(double(svm_c_rbf_optimized.IsSupportVector),1);

    disp("---------------RBF_Nested_Cross_Validation---------------")
    disp("RBF Accuracy Score: "+(sum(accuracy_score_rbf)/length(accuracy_score_rbf))/100);
    disp("RBF Best Hyperparameters for constraint: "+ hyperparam_constraint(best_hyperparam_rbf_constraint));
    disp("RBF Best Hyperparameters for sigma: "+ hyperparam_sigma(best_hyperparam_rbf_sigma));
    disp("RBF Support Vectors : " + support_vector_rbf + "(" + (support_vector_rbf/length(X)*100 + "%)"))
    
    best_hyperparam_poly_constraint = floor((mode(hyperparam_vector_poly))/length(q))+1;
    best_hyperparam_poly_q = mod(mode(hyperparam_vector_poly),length(q))+1;
    
    svm_c_poly_optimized = fitcsvm(X,y,"KernelFunction","polynomial","PolynomialOrder",hyperparam_sigma(best_hyperparam_poly_q),"BoxConstraint",hyperparam_constraint(best_hyperparam_poly_constraint));
    support_vector_poly = histc(double(svm_c_poly_optimized.IsSupportVector),1);

    disp("---------------Polynomial_Nested_Cross_Validation---------------")
    disp("Polynomial Accuracy Score: "+(sum(accuracy_score_poly)/length(accuracy_score_poly))/100);
    disp("Polynomial Best Hyperparameters for constraint: "+ hyperparam_constraint(best_hyperparam_poly_constraint));
    disp("Polynomial Best Hyperparameters for q: "+ hyperparam_q(best_hyperparam_poly_q));
    disp("Polynomial Support Vectors : " + support_vector_poly + "(" + (support_vector_poly/length(X)*100 + "%)"))
    
    tempC = containers.Map();
    tempC("c_rbf") = hyperparam_constraint(best_hyperparam_rbf_constraint);
    tempC("sigma_rbf") = hyperparam_sigma(best_hyperparam_rbf_sigma);
    tempC("sv_rbf") = support_vector_rbf;
    tempC("sv_rbf_%") = support_vector_rbf/length(X)*100;
    
    tempC("c_poly") = hyperparam_constraint(best_hyperparam_poly_constraint);
    tempC("q_poly") = hyperparam_q(best_hyperparam_poly_q);
    tempC("sv_poly") = support_vector_poly;
    tempC("sv_poly_%") = support_vector_poly/length(X)*100;

    bestParams = tempC;
end
