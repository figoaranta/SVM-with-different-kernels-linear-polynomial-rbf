load("gender.mat");

x_train = full(X(1:10, :));
y_train = Y(1:10,1);

data = [x_train, y_train];
length_of_unique = length(unique(X{:,1}));
sorted_data = sortrows(data,3,"descend");
sorted_data(:,1).Name = lower(sorted_data(:,1).Name);

% Remove any duplicate name and get the one with higher probability
hm = containers.Map;

i=1;
while (i~=size(sorted_data(:,1),1))
     if isKey(hm,(sorted_data(i,1).Name))
         sorted_data(i,:)=[];
         i = i - 1;
     else
         hm(string(sorted_data(i,1).Name))=1;
     end
     i = i + 1;
end

% Remove any special character
for i=1:length(sorted_data(:,1).Name)
    sorted_data(:,1).Name(i) = cellstr(regexprep(string(sorted_data(:,1).Name(i)),'[^a-zA-Z]',''));
end

% Shuffle sorted data
sorted_data = sorted_data(randperm(size(sorted_data(:,1),1)), :);

% Mapping each character to number
alphabet = 'abcdefghijklmnopqrstuvwxyz';
alphabet_map = containers.Map;
alphabet_map(' ')=0;
for i = 1:length(alphabet)
    alphabet_map(alphabet(i))=i;
end


% Getting the lengthiest name
names_length=cellfun(@(x) numel(x),sorted_data(:,1).Name);
longest_name=sorted_data(:,1).Name(names_length==max(names_length));
longest_name_length = max(names_length);


% Preprocessing Part
x_train = sorted_data(:,1);
x_train = x_train{:,:};
x_train = pad(x_train);

y_train = sorted_data(:,4);
y_train = y_train{:,:};


final_x_train = [];
final_y_train = [];

for i=1:length(x_train)
    vector_name = [];
    char_name = char(x_train(i));
    for j = 1:length(char_name)
%         disp(char_name(j));
        vector_name = [vector_name ; alphabet_map(char_name(j))];
    end
%     disp(vector_name');
    final_x_train = [final_x_train ; vector_name'];
end

for i=1:length(y_train)
    if char(y_train(i)) == 'M'
        final_y_train = [final_y_train ; 1];
    else
        final_y_train = [final_y_train ; 0];
    end
end

% Model Building
Mdl = fitcsvm(final_x_train,final_y_train, 'KernelFunction','linear', 'BoxConstraint',1);

name_to_predict = 'asdsad';
x_test = nameToVec(name_to_predict,longest_name_length);
label = predict( Mdl , x_test );

pred_label = containers.Map;
pred_label('1') = 'M';
pred_label('0') = 'F';
disp("Prediction for the name "+ name_to_predict + " is: " +pred_label(string(label)));

save('trained_svm.mat','Mdl');
