function [w, E] = OfflineLearning(x, d, f, gradf, lr, stopf)

[~, n] = size(x);
w = randn(n, size(d, 2));
epoch = 0;
E = [];
while true
    v = x * w;
    y = f(v);
    e = y - d;
    g = x' * (e .* gradf(v));
    w = w - lr * g;
    E(end+1) = sum(e(:).^2);
    if stopf(E, epoch)
        break;
    end
    fprintf('%.4f\n', E(end));
    epoch = epoch + 1;
end