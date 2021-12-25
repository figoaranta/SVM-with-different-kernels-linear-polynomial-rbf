% function bestParams = nested_cross_validation(X,y,outerFold,innerFold)
load("final_gender.mat");
X = X_gender(1:10000,:);
y = y_gender(1:10000,:);

hyperparam = containers.Map();

hyperparam("c")= {1,2,3};
hyperparam("sigma")= {1,2,3};
hyperparam("q")= {1,3,5};

outerLoop = 5;
innerLoop = 2;

length_of_fold = floor(length(X)/outerLoop);

hyperparam_vector_rbf = [];
cr_score_rbf = [];

hyperparam_vector_poly =[];
cr_score_poly = [];

enableRBF = true;
enablePoly = true;
findSupportVector = true;

for i=1:outerLoop
    x_test = X((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold , :);
    y_test = y((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold);

    x_train = X;
    x_train((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold , :) = [];
    y_train = y;
    y_train((outerLoop-i)*length_of_fold+1 : (outerLoop-i+1)*length_of_fold) = [];

    length_of_inner_fold = floor(length(x_train)/innerLoop);

    sigma = cell2mat(hyperparam("sigma"));
    c = cell2mat(hyperparam("c"));
    q = cell2mat(hyperparam("q"));
    
    cr_training_subset_1_rbf = [];
    cr_training_subset_2_rbf = [];

    cr_training_subset_1_poly = [];
    cr_training_subset_2_poly = [];

    cr_training_subset_1_li = [];
    cr_training_subset_2_li = [];

    for j=1:innerLoop
        x_validation_set = x_train((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: );
        y_validation_set = y_train((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: );
    
        x_training_subset = x_train;
        x_training_subset((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: ) = [];
        y_training_subset = y_train;
        y_training_subset((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: ) = [];
        
        
    
        for k = 1:length(c)
            c_inner = c(k);
    
%             RBF
        if(enableRBF)
            for l = 1:length(sigma)
                sigma_inner = sigma(l);
                    
                svm_c_rbf = fitcsvm(x_training_subset,y_training_subset,"KernelFunction","rbf","KernelScale",sigma_inner,"BoxConstraint",c_inner);
                
                yhat_rbf = predict(svm_c_rbf,x_validation_set);
                
                cr_inner_rbf = ( sum(yhat_rbf==1 & y_validation_set==1) + sum(yhat_rbf==0 & y_validation_set==0)) /length(x_validation_set)*100 ;
                
                if j ==1          
                    cr_training_subset_1_rbf = [cr_training_subset_1_rbf,cr_inner_rbf];
                else
                    cr_training_subset_2_rbf = [cr_training_subset_2_rbf,cr_inner_rbf];
                end
               
            end
        end

%             Poly
        if(enablePoly)
            for m = 1:length(q)
                q_inner = q(m);
        
                svm_c_poly = fitcsvm(x_training_subset,y_training_subset,"KernelFunction","polynomial","PolynomialOrder",q_inner,"BoxConstraint",c_inner);
                
                yhat_poly = predict(svm_c_poly,x_validation_set);
            
                cr_inner_rbf = (sum(yhat_poly==1 & y_validation_set==1) + sum(yhat_poly==0 & y_validation_set==0))/length(x_validation_set)*100 ;
                
                if j ==1
                    cr_training_subset_1_poly = [cr_training_subset_1_poly,cr_inner_rbf];
                else
                    cr_training_subset_2_poly = [cr_training_subset_2_poly,cr_inner_rbf];
                end

            end
        end

        end
    end 
%     End of inner loop

%     Best c and sigma for RBF
    if(enableRBF)
        combined_cr_t1_t2_rbf = cr_training_subset_1_rbf + cr_training_subset_2_rbf;
        [best_hyperparam_score_rbf,best_hyperparam_rbf] = max(combined_cr_t1_t2_rbf);
        hyperparam_vector_rbf = [hyperparam_vector_rbf,best_hyperparam_rbf];
    
        best_hyperparam_rbf_c = ceil((best_hyperparam_rbf)/length(sigma));
        best_hyperparam_rbf_sigma = mod(best_hyperparam_rbf,length(sigma));
        if(best_hyperparam_rbf_sigma == 0)
            best_hyperparam_rbf_sigma = best_hyperparam_rbf_sigma+length(sigma);
        end
    end


%     Best c and q for Poly
    if(enablePoly)
        combined_cr_t1_t2_poly = cr_training_subset_1_poly + cr_training_subset_2_poly;
        [best_hyperparam_score_poly,best_hyperparam_poly] = max(combined_cr_t1_t2_poly);
        hyperparam_vector_poly = [hyperparam_vector_poly,best_hyperparam_poly];
       
        best_hyperparam_poly_c = ceil((best_hyperparam_poly)/length(q));
        best_hyperparam_poly_q = mod(best_hyperparam_poly,length(q));
        if(best_hyperparam_poly_q == 0)
            best_hyperparam_poly_q = best_hyperparam_poly_q+length(q);
        end
    end
   

%     Value of best c and sigma for RBF Model
    if(enableRBF)
        c_outer_rbf = c(best_hyperparam_rbf_c);
        sigma_outer_rbf = sigma(best_hyperparam_rbf_sigma);
    
        svm_c_rbf = fitcsvm(x_train,y_train,"KernelFunction","rbf","KernelScale",sigma_outer_rbf,"BoxConstraint",c_outer_rbf);
        yhat_rbf = predict(svm_c_rbf,x_test);
    
        cr_outer_rbf = (sum((yhat_rbf==1 & y_test==1)) + sum((yhat_rbf==0 & y_test==0)))/length(x_test)*100 ;
        cr_score_rbf = [cr_score_rbf,cr_outer_rbf ];
    end
    
%     Value of best c and q for Poly Model
    if(enablePoly)
        c_outer_poly = c(best_hyperparam_poly_c);
        q_outer_poly = q(best_hyperparam_poly_q);
    
        svm_c_poly = fitcsvm(x_train,y_train,"KernelFunction","polynomial","PolynomialOrder",q_outer_poly,"BoxConstraint",c_outer_poly);
        yhat_poly = predict(svm_c_poly,x_test);
    
        cr_outer_poly = ( sum((yhat_poly==1 & y_test==1)) + sum((yhat_poly==0 & y_test==0)))/length(x_test)*100 ;
        cr_score_poly = [cr_score_poly,cr_outer_poly ];
    end

end
% End of outer loop

hyperparam_c = cell2mat(hyperparam("c"));
hyperparam_sigma = cell2mat(hyperparam("sigma"));
hyperparam_q = cell2mat(hyperparam("q"));

% RBF Summary
if(enableRBF)
    best_hyperparam_rbf_c = ceil((mode(hyperparam_vector_rbf))/length(sigma));
    best_hyperparam_rbf_sigma = mod(mode(hyperparam_vector_rbf),length(sigma));
    if(best_hyperparam_rbf_sigma == 0)
        best_hyperparam_rbf_sigma = best_hyperparam_rbf_sigma+length(sigma);
    end
    
    disp("RBF Classification Rate Score: "+(sum(cr_score_rbf)/length(cr_score_rbf)));
    disp("RBF Best Hyperparameters for c: "+ hyperparam_c(best_hyperparam_rbf_c));
    disp("RBF Best Hyperparameters for sigma: "+ hyperparam_sigma(best_hyperparam_rbf_sigma));

%     Find Support Vector
    if(findSupportVector)
        c_svm_rbf = fitcsvm(X,y,"KernelScale",hyperparam_sigma(best_hyperparam_rbf_sigma),"BoxConstraint",hyperparam_c(best_hyperparam_rbf_c));
        rbf_sv = size(c_svm_rbf.SupportVectors,1);
        disp("Number of Support Vector for RBF Model: "+ rbf_sv);
    end
end

% Poly Summary
if(enablePoly)
    best_hyperparam_poly_c = ceil((mode(hyperparam_vector_poly))/length(q));
    best_hyperparam_poly_q = mod(mode(hyperparam_vector_poly),length(q));
    if(best_hyperparam_poly_q == 0)
        best_hyperparam_poly_q = best_hyperparam_poly_q+length(q);
    end
    
    disp("Poly Classification Rate Score: "+(sum(cr_score_poly)/length(cr_score_poly)));
    disp("Poly Best Hyperparameters for c: "+ hyperparam_c(best_hyperparam_poly_c));
    disp("Poly Best Hyperparameters for q: "+ hyperparam_q(best_hyperparam_poly_q));

    if(findSupportVector)
        c_svm_poly = fitcsvm(X,y,"KernelFunction","polynomial","PolynomialOrder",hyperparam_q(best_hyperparam_poly_q),"BoxConstraint",hyperparam_c(best_hyperparam_poly_c));
        poly_sv = size(c_svm_poly.SupportVectors,1);
        disp("Number of Support Vector for Poly Model: "+ poly_sv);
    end
end

%     tempC = containers.Map();
%     tempC("epsilon_rbf") = hyperparam_epsilon(best_hyperparam_rbf_epsilon);
%     tempC("sigma_rbf") = hyperparam_sigma(best_hyperparam_rbf_sigma);
%     tempC("sv_rbf") = rbf_sv;
%     tempC("epsilon_poly") = hyperparam_epsilon(best_hyperparam_poly_epsilon);
%     tempC("q_poly") = hyperparam_q(best_hyperparam_poly_q);
%     tempC("sv_poly") = poly_sv;
% 
%     bestParams = tempC;

% end


