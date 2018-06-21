function list = find_bad_files(folder_name)
    list = [];

    files = dir(strcat(folder_name, '/*.wav'));
    for file = files';
        file.name
        try
           mirfeatures(strcat(folder_name, '/' , file.name));
        catch
           list = [list; file.name]
            
    end

end
