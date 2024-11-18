### Explanation for File Recursion Problem

#### Efficiency

**Time Complexity:**
- The time complexity is \(O(n)\), where \(n\) is the total number of files and directories in the given path. This is because the algorithm has to visit each file and directory exactly once to determine its type and whether it matches the suffix.

**Space Complexity:**
- The space complexity includes the space required for the call stack due to recursion, which can go as deep as the directory depth \(d\). Thus, the worst-case space complexity is \(O(d)\), where \(d\) is the maximum depth of the directory tree.
- Additional space is used for storing the result list `found_files`, which in the worst case can be \(O(f)\), where \(f\) is the number of matching files.

#### Code Design

- **Algorithm Choice:**
  - The problem is approached using recursion to handle the potentially unlimited depth of subdirectories. This is a natural choice for tree-like structures such as file systems.
  - The recursive function `recursive_search` traverses the directory tree, checking whether each item is a file or a directory. If it's a directory, the function calls itself recursively. If it's a file that matches the suffix, it's added to the result list.

- **Data Structure Choice:**
  - The choice of using a list `found_files` to accumulate the results is straightforward and efficient for this purpose.
  - Use of Python's built-in `os` module functions like `os.listdir()`, `os.path.join()`, `os.path.isdir()`, and `os.path.isfile()` provides a robust and cross-platform way to interact with the filesystem.

#### Readability

- The code uses proper Python conventions and is well-documented with comments explaining the purpose of the function and its arguments.
- Variable names are descriptive and clearly indicate their purpose (`suffix`, `path`, `found_files`, `recursive_search`).
- The recursive approach is well-structured and handles edge cases like non-existent directories and empty directories gracefully.
