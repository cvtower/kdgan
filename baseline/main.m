fprintf('hello, world!\n');

num_row = 2;
num_col = 3;
fc7 = zeros(num_row, num_col, 'single');

for r = 1:num_row
    for c = 1:num_col
        fc7(r,c) = 1/(r+c-1);
    end
end

save feature.mat fc7

exit;