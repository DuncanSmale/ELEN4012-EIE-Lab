function H = LoadMatrix(textFile,N,M, save_name)
Ht = fileread(textFile);

H = ReshapeAndConvertToBinary(Ht,N,M);

save(save_name, 'H');
end
