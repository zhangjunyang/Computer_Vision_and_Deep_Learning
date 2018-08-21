%% 采用MCR判定聚类效果
 B = class(:,4);
 B = reshape(B,1,row);
 A = [ones(1,100),2 * ones(1,100),3 *ones(1,100),4 * ones(1,100)];
 
sum = 0;
for i = 1:row
    if ( A(1,i) ~= B(1,i))
        sum = sum + 1;
    end
end
MCR = sum / row;
fprintf('MCR = %d\n',MCR);
