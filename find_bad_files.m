%{
Filename: find_bad_files.m

Tests every audio file for whether it works with the mirfeatures function and
returns a list of files which do not work.
%}

function list = find_bad_files(folder_name)
		%{
		Tests every audio file for whether it works with the mirfeatures function
		and returns a list of files which do not work.

		Input:
				folder_name:
					name of the map containing the audio files

		Output:
				A list of audio files which do not work with mirfeatures

		%}
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
