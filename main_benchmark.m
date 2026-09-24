clc; clear; close all;

% datasets and Parameters
All_datasets = {'Yeast', 'Parkinson', 'Heart', 'CTO', 'BCWD'};
balancing_methods = {'None', 'SMOTE', 'G-SMOTE'};
classifiers = {'Standard', 'Modified', 'Fuzzy', 'Radius'};
k_values = 1:10;

all_results = table();

for d = 1:length(All_datasets)
    dataset = All_datasets{d};
    fprintf('\n====== Veri Seti: %s ======\n', dataset);
    
    % Data pre-processing
    data_table = readtable([dataset, '.xlsx'], 'Sheet', dataset);
    X1 = table2array(data_table(:, 1:end-1));
    Y1 = table2array(data_table(:, end));
    
    % Min-Max Norm.
    X_normalized = (X1 - min(X1)) ./ (max(X1) - min(X1) + 1e-10);
    
    % %80 Train, %20 Test
    rng(42); 
    cv_part = cvpartition(Y1, 'HoldOut', 0.20);
    X_train = X_normalized(training(cv_part), :);
    Y_train = Y1(training(cv_part), :);
    X_test = X_normalized(test(cv_part), :);
    Y_test = Y1(test(cv_part), :);
    
    % Class Identification
    classes = unique(Y_train);
    counts = [sum(Y_train == classes(1)), sum(Y_train == classes(2))];
    [~, min_idx] = min(counts);
    minority_class = classes(min_idx);
    [~, max_idx] = max(counts);
    majority_class = classes(max_idx);

    % Balancing Loop
    for b = 1:length(balancing_methods)
        balancing_type = balancing_methods{b};
        
        % Classifier Loop
        for c = 1:length(classifiers)
            curr_classifier = classifiers{c};
            fprintf('İşleniyor: %s + %s\n', balancing_type, curr_classifier);
            
            % 5-Fold Cross Validation and choosing k
            cv_kfold = cvpartition(Y_train, 'KFold', 5);
            k_errors = zeros(length(k_values), 1);
            
            for k_idx = 1:length(k_values)
                k = k_values(k_idx);
                fold_err = zeros(5, 1);
                
                for f = 1:5
                    X_train_f = X_train(training(cv_kfold, f), :);
                    Y_train_f = Y_train(training(cv_kfold, f));
                    X_test_f = X_train(test(cv_kfold, f), :);
                    Y_test_f = Y_train(test(cv_kfold, f));
                    
                    % Balance the Train Data
                    if strcmp(balancing_type, 'SMOTE')
                        [X_bal, Y_bal] = smote_custom(X_train_f, Y_train_f, 5, minority_class, majority_class);
                    elseif strcmp(balancing_type, 'G-SMOTE')
                        [X_bal, Y_bal] = g_smote_custom(X_train_f, Y_train_f, 0.1, minority_class, majority_class);
                    else
                        X_bal = X_train_f; Y_bal = Y_train_f;
                    end
                    
                    % Prediction
                    preds = run_classifier(X_bal, Y_bal, X_test_f, k, curr_classifier, classes);
                    fold_err(f) = sum(preds ~= Y_test_f) / length(Y_test_f);
                end
                k_errors(k_idx) = mean(fold_err);
            end
            
            [~, best_idx] = min(k_errors);
            best_k = k_values(best_idx);
            
            if strcmp(balancing_type, 'SMOTE')
                [X_final_train, Y_final_train] = smote_custom(X_train, Y_train, 5, minority_class, majority_class);
            elseif strcmp(balancing_type, 'G-SMOTE')
                [X_final_train, Y_final_train] = g_smote_custom(X_train, Y_train, 0.1, minority_class, majority_class);
            else
                X_final_train = X_train; Y_final_train = Y_train;
            end
            
            final_preds = run_classifier(X_final_train, Y_final_train, X_test, best_k, curr_classifier, classes);
            
            % performance measures
            overall_acc = sum(final_preds == Y_test) / length(Y_test);
            min_mask = (Y_test == minority_class);
            min_acc = sum(final_preds(min_mask) == Y_test(min_mask)) / sum(min_mask);
            maj_mask = (Y_test == majority_class);
            maj_acc = sum(final_preds(maj_mask) == Y_test(maj_mask)) / sum(maj_mask);
            
            res_row = {dataset, balancing_type, curr_classifier, best_k, overall_acc, min_acc, maj_acc};
            all_results = [all_results; res_row];
        end
    end
end

all_results.Properties.VariableNames = {'Dataset', 'Balancing', 'Classifier', 'Best_k', 'Overall_Acc', 'Minority_Acc', 'Majority_Acc'};
disp(all_results);

%% Functions

function p = run_classifier(X_tr, Y_tr, X_te, k, type, classes)
    switch type
        case 'Standard', p = knn_standard(X_tr, Y_tr, X_te, k);
        case 'Modified', p = knn_modified(X_tr, Y_tr, X_te, k, classes);
        case 'Fuzzy',    p = knn_fuzzy(X_tr, Y_tr, X_te, k, 2, classes);
        case 'Radius',   p = knn_radius(X_tr, Y_tr, X_te, 0.3); 
    end
end

function [X_balanced, Y_balanced] = smote_custom(X_train, Y_train, k_neighbors, min_cl, maj_cl)
    X_min = X_train(Y_train == min_cl, :);
    n_min = size(X_min, 1);
    n_maj = sum(Y_train == maj_cl);
    
    % minorty chechk
    if n_min < 2
        X_balanced = X_train; Y_balanced = Y_train; return;
    end
    
    num_to_generate = n_maj - n_min;
    X_synthetic = zeros(num_to_generate, size(X_train, 2));
    k_safe = min(k_neighbors, n_min - 1);
    
    for i = 1:num_to_generate
        idx = randi(n_min);
        sample = X_min(idx, :);
        dists = sqrt(sum((X_min - sample).^2, 2));
        [~, sorted_idx] = sort(dists);
        neighbor_idx = sorted_idx(randi([2, k_safe + 1]));
        X_synthetic(i, :) = sample + rand() * (X_min(neighbor_idx, :) - sample);
    end
    X_balanced = [X_train; X_synthetic];
    Y_balanced = [Y_train; repmat(min_cl, num_to_generate, 1)];
end

function [X_balanced, Y_balanced] = g_smote_custom(X_train, Y_train, gamma, min_cl, maj_cl)
    X_min = X_train(Y_train == min_cl, :);
    n_min = size(X_min, 1);
    if n_min < 1, X_balanced = X_train; Y_balanced = Y_train; return; end
    
    num_to_generate = sum(Y_train == maj_cl) - n_min;
    X_synthetic = zeros(num_to_generate, size(X_train, 2));
    
    for i = 1:num_to_generate
        center = X_min(randi(n_min), :);
        direction = randn(1, size(X_train, 2));
        direction = direction / (norm(direction) + 1e-10);
        X_synthetic(i, :) = center + (gamma * rand()) * direction;
    end
    X_balanced = [X_train; X_synthetic];
    Y_balanced = [Y_train; repmat(min_cl, num_to_generate, 1)];
end

function KNN = knn_standard(X_train, Y_train, X_test, k)
    n_test = size(X_test, 1);
    KNN = zeros(n_test, 1);
    for i = 1:n_test
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
        [~, sorted_indices] = sort(distances);
        KNN(i) = mode(Y_train(sorted_indices(1:k)));
    end
end

function m_KNN = knn_modified(X_train, Y_train, X_test, k, classes)
    n_test = size(X_test, 1);
    m_KNN = zeros(n_test, 1);
    for i = 1:n_test
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
        [sorted_dists, sorted_idx] = sort(distances);
        weights = 1 ./ (sorted_dists(1:k) + 1e-10);
        labels = Y_train(sorted_idx(1:k));
        w1 = sum(weights(labels == classes(1)));
        w2 = sum(weights(labels == classes(2)));
        if w1 >= w2, m_KNN(i) = classes(1); else, m_KNN(i) = classes(2); end
    end
end

function Fuzzy_KNN = knn_fuzzy(X_train, Y_train, X_test, k, m, classes)
    n_test = size(X_test, 1);
    Fuzzy_KNN = zeros(n_test, 1);
    power_val = 2 / (m - 1);
    for i = 1:n_test
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
        [sorted_dists, sorted_idx] = sort(distances);
        weights = 1 ./ (sorted_dists(1:k).^power_val + 1e-10);
        labels = Y_train(sorted_idx(1:k));
        m1 = sum(weights(labels == classes(1)));
        m2 = sum(weights(labels == classes(2)));
        if m1 >= m2, Fuzzy_KNN(i) = classes(1); else, Fuzzy_KNN(i) = classes(2); end
    end
end

function r_KNN = knn_radius(X_train, Y_train, X_test, gamma)
    n_test = size(X_test, 1);
    r_KNN = zeros(n_test, 1);
    for i = 1:n_test
        distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
        idx = find(distances <= gamma);
        if isempty(idx), [~, nearest] = min(distances); r_KNN(i) = Y_train(nearest);
        else, r_KNN(i) = mode(Y_train(idx)); end
    end
end
