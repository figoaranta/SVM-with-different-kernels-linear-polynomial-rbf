function bestParams = r_nested_cross_validation(X,y,outerFold,innerFold)

    hyperparam = containers.Map();
    
    hyperparam("epsilon")= {5,10,20};
    hyperparam("sigma")= {1,10,15};
    hyperparam("q")= {1,2,3,4,5};
    hyperparam("c")= {100,500,1000};
    
    outerLoop = outerFold;
    innerLoop = innerFold;
    
    length_of_fold = floor(length(X)/outerLoop);
    
    hyperparam_vector_rbf = [];
    rmse_score_rbf = [];
    
    hyperparam_vector_poly =[];
    rmse_score_poly = [];
    
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
        epsilon = cell2mat(hyperparam("epsilon"));
        q = cell2mat(hyperparam("q"));
        c = cell2mat(hyperparam("c"));
        
        rmse_training_subset_1_rbf = [];
        rmse_training_subset_2_rbf = [];
    
        rmse_training_subset_1_poly = [];
        rmse_training_subset_2_poly = [];
    
        for j=1:innerLoop
            x_validation_set = x_train((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: );
            y_validation_set = y_train((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: );
        
            x_training_subset = x_train;
            x_training_subset((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: ) = [];
            y_training_subset = y_train;
            y_training_subset((innerLoop - j)*length_of_inner_fold+1 : (innerLoop-j+1)*length_of_inner_fold ,: ) = [];
            
            
        
            for k = 1:length(epsilon)
                epsilon_inner = epsilon(k);

                for p=1:length(c)
                    c_inner = c(p);
                    
    
            %             RBF
                    if(enableRBF)
                        for l = 1:length(sigma)
                            sigma_inner = sigma(l);
                                
                            svm_r_rbf = fitrsvm(x_training_subset,y_training_subset,"KernelScale",sigma_inner,"Epsilon",epsilon_inner,"BoxConstraint",c_inner);
                            yhat_rbf = predict(svm_r_rbf,x_validation_set);
                
                            rmse_inner_rbf = sqrt(mean((y_validation_set - yhat_rbf).^2));
                            
                            if j ==1          
                                rmse_training_subset_1_rbf = [rmse_training_subset_1_rbf,rmse_inner_rbf];
                            else
                                rmse_training_subset_2_rbf = [rmse_training_subset_2_rbf,rmse_inner_rbf];
                            end
                           
                        end
                    end
            
            %             Poly
                    if(enablePoly)
                        for m = 1:length(q)
                            q_inner = q(m);
                    
                            svm_r_poly = fitrsvm(x_training_subset,y_training_subset,"KernelFunction","polynomial","PolynomialOrder",q_inner,"Epsilon",epsilon_inner,"BoxConstraint",c_inner);
                            
                            yhat_poly = predict(svm_r_poly,x_validation_set);
                        
                            rmse_inner_poly = sqrt(mean((y_validation_set - yhat_poly).^2));
                            
                            if j ==1
                                rmse_training_subset_1_poly = [rmse_training_subset_1_poly,rmse_inner_poly];
                            else
                                rmse_training_subset_2_poly = [rmse_training_subset_2_poly,rmse_inner_poly];
                            end
            
                        end
                    end
                end
            end
        end 
    %     End of inner loop
    
    %     Best epsilon and sigma for RBF
        if(enableRBF)
            combined_rmse_t1_t2_rbf = rmse_training_subset_1_rbf + rmse_training_subset_2_rbf;
            [best_hyperparam_score_rbf,best_hyperparam_rbf] = min(combined_rmse_t1_t2_rbf);
            best_hyperparam_rbf = best_hyperparam_rbf -1;
            hyperparam_vector_rbf = [hyperparam_vector_rbf,best_hyperparam_rbf];
            

            best_hyperparam_rbf_epsilon = floor((best_hyperparam_rbf )/(length(c)*length(sigma)))+1;
            best_hyperparam_rbf_c = mod(floor((best_hyperparam_rbf )/length(sigma)) , length(c))+1;
            best_hyperparam_rbf_sigma = mod(best_hyperparam_rbf,length(sigma))+1;

        end
    
    
    %     Best epsilon and q for Poly
        if(enablePoly)
            combined_rmse_t1_t2_poly = rmse_training_subset_1_poly + rmse_training_subset_2_poly;
            [best_hyperparam_score_poly,best_hyperparam_poly] = min(combined_rmse_t1_t2_poly);
            best_hyperparam_poly = best_hyperparam_poly-1;
            hyperparam_vector_poly = [hyperparam_vector_poly,best_hyperparam_poly];
           
            best_hyperparam_poly_epsilon = floor((best_hyperparam_poly )/(length(c)*length(q)))+1;
            best_hyperparam_poly_c = mod(floor((best_hyperparam_poly )/length(q)),length(c))+1;
            best_hyperparam_poly_q = mod(best_hyperparam_poly,length(q))+1;

        end
       
    
    %     Value of best epsilon and sigma for RBF Model
        if(enableRBF)
            epsilon_outer_rbf = epsilon(best_hyperparam_rbf_epsilon);
            sigma_outer_rbf = sigma(best_hyperparam_rbf_sigma);
            c_outer_rbf = c(best_hyperparam_rbf_c);
        
            svm_r_rbf = fitrsvm(x_train,y_train,"KernelScale",sigma_outer_rbf,"Epsilon",epsilon_outer_rbf,"BoxConstraint",c_outer_rbf);
            yhat_rbf = predict(svm_r_rbf,x_test);
        
            rmse_outer_rbf = sqrt(mean((y_test - yhat_rbf).^2));
            rmse_score_rbf = [rmse_score_rbf,rmse_outer_rbf ];
        end

%         best_hyperparam_rbf=best_hyperparam_rbf+1;
%         best_hyperparam_poly=best_hyperparam_poly+1;
        
    %     Value of best epsilon and q for Poly Model
        if(enablePoly)
            epsilon_outer_poly = epsilon(best_hyperparam_poly_epsilon);
            q_outer_poly = q(best_hyperparam_poly_q);
            c_outer_poly = c(best_hyperparam_poly_c);

            svm_r_poly = fitrsvm(x_train,y_train,"KernelFunction","polynomial","PolynomialOrder",q_outer_poly,"Epsilon",epsilon_outer_poly,"BoxConstraint",c_outer_poly);
            yhat_poly = predict(svm_r_poly,x_test);
        
            rmse_outer_poly = sqrt(mean((y_test - yhat_poly).^2));
            rmse_score_poly = [rmse_score_poly,rmse_outer_poly ];
        end
    
    end
    % End of outer loop
    
    hyperparam_epsilon = cell2mat(hyperparam("epsilon"));
    hyperparam_sigma = cell2mat(hyperparam("sigma"));
    hyperparam_q = cell2mat(hyperparam("q"));
    hyperparam_c = cell2mat(hyperparam("c"));
     
    % RBF Summary
    if(enableRBF) 
        best_hyperparam_rbf_epsilon = floor(mode(hyperparam_vector_rbf) /(length(c)*length(sigma)))+1;
        best_hyperparam_rbf_c = mod(floor(mode(hyperparam_vector_rbf) /length(sigma)),length(c))+1;
        best_hyperparam_rbf_sigma = mod(mode(hyperparam_vector_rbf),length(sigma))+1;
        
        disp("RBF RMSE Score: "+(sum(rmse_score_rbf)/length(rmse_score_rbf)));
        disp("RBF Best Hyperparameters for epsilon: "+ hyperparam_epsilon(best_hyperparam_rbf_epsilon));
        disp("RBF Best Hyperparameters for sigma: "+ hyperparam_sigma(best_hyperparam_rbf_sigma));
        disp("RBF Best Hyperparameters for c: "+ hyperparam_c(best_hyperparam_rbf_c));
    
    %     Find Support Vector
        if(findSupportVector)
            r_svm_rbf = fitrsvm(X,y,"KernelScale",hyperparam_sigma(best_hyperparam_rbf_sigma),"Epsilon",hyperparam_epsilon(best_hyperparam_rbf_epsilon),"BoxConstraint",hyperparam_c(best_hyperparam_rbf_c));
            rbf_sv = size(r_svm_rbf.SupportVectors,1);
            disp("Number of Support Vector for RBF Model: "+ rbf_sv + " " +(rbf_sv/length(X)*100)+ "%");
        end
    end
    
    % Poly Summary
    if(enablePoly)
        best_hyperparam_poly_epsilon = floor(mode(hyperparam_vector_poly) /(length(c)*length(q)))+1;
        best_hyperparam_poly_c = mod(floor((mode(hyperparam_vector_poly) )/length(q)),length(c))+1;
        best_hyperparam_poly_q = mod(mode(hyperparam_vector_poly),length(q))+1;
        if(best_hyperparam_poly_q == 0)
            best_hyperparam_poly_q = best_hyperparam_poly_q+length(q);
        end
        
        disp("Poly RMSE Score: "+(sum(rmse_score_poly)/length(rmse_score_poly)));
        disp("Poly Best Hyperparameters for epsilon: "+ hyperparam_epsilon(best_hyperparam_poly_epsilon));
        disp("Poly Best Hyperparameters for q: "+ hyperparam_q(best_hyperparam_poly_q));
        disp("Poly Best Hyperparameters for c: "+ hyperparam_c(best_hyperparam_poly_c));
    
        if(findSupportVector)
            r_svm_poly = fitrsvm(X,y,"KernelFunction","polynomial","PolynomialOrder",hyperparam_q(best_hyperparam_poly_q),"Epsilon",hyperparam_epsilon(best_hyperparam_poly_epsilon),"BoxConstraint",hyperparam_c(best_hyperparam_poly_c));
            poly_sv = size(r_svm_poly.SupportVectors,1);
            disp("Number of Support Vector for Poly Model: "+ poly_sv + " " +(poly_sv/length(X)*100) + "%");
        end
    end

    tempC = containers.Map();
    tempC("c_rbf") = hyperparam_c(best_hyperparam_rbf_c);
    tempC("epsilon_rbf") = hyperparam_epsilon(best_hyperparam_rbf_epsilon);
    tempC("sigma_rbf") = hyperparam_sigma(best_hyperparam_rbf_sigma);
    tempC("sv_rbf") = rbf_sv;

    tempC("c_poly") = hyperparam_c(best_hyperparam_poly_c);
    tempC("epsilon_poly") = hyperparam_epsilon(best_hyperparam_poly_epsilon);
    tempC("q_poly") = hyperparam_q(best_hyperparam_poly_q);
    tempC("sv_poly") = poly_sv;

    bestParams = tempC;
end