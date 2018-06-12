function list = find_bad_files(map_name)
    list = [];

    files = dir(strcat(map_name, '/*.wav'));
    for file = files';
        file.name
        try
           mirfeatures(strcat(map_name, '/' , file.name));
        catch
           list = [list; file.name]
            
    end

end

