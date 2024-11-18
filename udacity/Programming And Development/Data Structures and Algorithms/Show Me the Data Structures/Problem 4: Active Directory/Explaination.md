### Explanation

#### Efficiency

**Time Complexity:**
- The time complexity is \(O(n)\), where \(n\) is the total number of users and sub-groups within the given group. This is because in the worst case, each user and group within the group hierarchy needs to be checked exactly once.

**Space Complexity:**
- The space complexity is \(O(d)\), where \(d\) is the maximum depth of the group hierarchy. This is due to the recursive stack space used in the function calls.

#### Code Design

- **Group Class:**
  - Encapsulates the properties and behaviors of a group, including managing sub-groups and users.
  - Provides methods to add sub-groups and users, and to retrieve these elements.
  
- **Recursive User Membership Check:**
  - The `is_user_in_group` function uses recursion to check if the user is in the current group or any of its sub-groups.
  - If the user is found in the current group's user list, it returns `True`.
  - If not, it recursively checks each sub-group.

#### Readability

- The code is written in clear and concise Python, following proper naming conventions and structured logically.
- The methods in the `Group` class are well-named and intuitive for their purposes.
- The `is_user_in_group` function is straightforward, with conditions and recursive calls clearly explained.
