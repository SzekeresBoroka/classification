function [w,E] = OnlineLearning(x, d, f, gradf, lr, stop)
[N, n] = size(x);
 w = randn(n,size(d,2));
 epoch = 0;
 while true
     E = 0;
     for i = randperm(N)
     vi = x(i,:)*w;
     yi = f(vi);
    ei = yi-d(i,:); % using square loss function
     gi = x(i,:)' * ei .* gradf(vi); %
     w = w - lr * gi;
     E = E + sum(ei.^2);
     end
     if stop(E, epoch), break; end
     epoch = epoch + 1;
 end
