function [output_vector] = nameToVec(input_name,longest_name_length)

% Mapping each character to number
alphabet = 'abcdefghijklmnopqrstuvwxyz';
alphabet_map = containers.Map;
alphabet_map(' ')=0;
for i = 1:length(alphabet)
    alphabet_map(alphabet(i))=i;
end

% Convert name to vector
lower_test = lower(input_name);
pad_test = char(pad(lower_test,longest_name_length));
test_vector = [];
for i=1:length(pad_test)
    test_vector = [test_vector ; alphabet_map(pad_test(i))];
end
test_vector = test_vector';

output_vector = test_vector;
end

