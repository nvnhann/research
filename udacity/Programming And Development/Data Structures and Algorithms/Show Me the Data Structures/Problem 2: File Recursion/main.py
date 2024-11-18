import os


def find_files(suffix, path):
    """
    Find all files beneath path with file name suffix.

    Note that a path may contain further subdirectories
    and those subdirectories may also contain further subdirectories.

    There are no limit to the depth of the subdirectories can be.

    Args:
      suffix(str): suffix if the file name to be found
      path(str): path of the file system

    Returns:
       a list of paths
    """
    if not os.path.isdir(path):
        return []

    found_files = []

    def recursive_search(current_path):
        for item in os.listdir(current_path):
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                recursive_search(full_path)
            elif os.path.isfile(full_path) and item.endswith(suffix):
                found_files.append(full_path)

    recursive_search(path)

    return found_files


# Test cases
print("Test Case 1")
print(find_files(".c", "testdir"))  # Expected: all .c files in the directory structure

print("Test Case 2 (Empty directory)")
os.mkdir('./test_empty_dir')
print(find_files(".c", "./test_empty_dir"))  # Expected: []

print("Test Case 3 (Non-existent directory)")
print(find_files(".c", "./non_existent_dir"))  # Expected: []

# Clean up the empty directory created for test case 2
os.rmdir('./test_empty_dir')
