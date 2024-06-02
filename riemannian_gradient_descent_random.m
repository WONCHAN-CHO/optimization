function riemannian_gradient_descent_random
    % Parameters
    initial_point = rand(1, 2) * 20 - 10;  % Random initial point in the range [-10, 10]
    x_star = rand(1, 2) * 20 - 10;        % Random target point in the range [-10, 10]
    alpha = 0.001;  % Initial learning rate
    epsilon = 1e-2;
    max_iter = 1000000;
    tolerance = 0;

    % Run Riemannian Gradient Descent
    [converged_point, distances] = RGD(initial_point, x_star, alpha, epsilon, max_iter, tolerance);

    % Print results
    disp('Initial point:');
    disp(initial_point);
    disp('Target point (x*):');
    disp(x_star);
    disp('Converged point:');
    disp(converged_point);
    disp('Final distance to x*:');
    disp(distances(end));

    % Plot distance to x* over iterations in log scale
    figure;
    semilogy(distances, 'r-');  % Only red line without markers
    xlabel('Iteration');
    ylabel('Distance to x* (log scale)');
    title('Distance to x* over iterations (Riemannian Gradient Descent)');
    grid on;

    % Display the values of the unknowns
    fprintf('Initial point: [%.4f, %.4f]\n', initial_point(1), initial_point(2));
    fprintf('Target point (x*): [%.4f, %.4f]\n', x_star(1), x_star(2));
end

function [x_k, distances] = RGD(x0, x_star, alpha, epsilon, max_iter, tolerance)
    % Helper functions
    compute_gradient = @(x) 2 * (x - x_star);
    retraction = @(x, v) x + v;
    distance = @(x, y) norm(x - y);
    
    k = 0;
    x_k = x0;
    distances = distance(x_k, x_star);
    initial_alpha = alpha;
    
    while k < max_iter
        g_mu = compute_gradient(x_k);
        alpha = initial_alpha / (1 + 0.001 * k);  % Gradually decreasing learning rate
        x_k1 = retraction(x_k, -alpha * g_mu);
        
        distances = [distances; distance(x_k1, x_star)];
        
        if norm(g_mu) < epsilon
            fprintf('Stopping criterion met at iteration %d: gradient norm < epsilon\n', k);
            break;
        end
        
        if distances(end) < tolerance
            fprintf('Stopping criterion met at iteration %d: distance to x* is below tolerance\n', k);
            break;
        end
        
        x_k = x_k1;
        k = k + 1;
    end
end
