### Explanation:

#### Efficiency:
The provided solution sorts the given list in \(O(n)\) time complexity and with \(O(1)\) space complexity by using a single traversal. This meets the requirement to sort the array in a single traversal.

#### Breakdown of Efficiency:
- **Single Pass Traversal**: By using three pointers (`low`, `mid`, and `high`), the algorithm achieves sorting in a single pass through the list. The `mid` pointer is used to examine elements, and based on their values, appropriate swaps are performed to ensure they are moved to their correct positions.
- **Swap Operations**: Each swap operation is constant time, \(O(1)\), and as such, doesn't contribute additional complexity beyond the single traversal.

#### Design Choices:
- **Three-Way Partitioning**: The use of three pointers (`low`, `mid`, `high`) effectively partitions the list into three regions:
  - **0 to `low` - 1**: All 0s.
  - **`low` to `mid` - 1**: All 1s.
  - **`mid` to `high`**: Unprocessed elements.
  - **`high` + 1 to End**: All 2s.
  This design ensures in-place sorting without requiring additional space.
  
- **Conditional Checks and Swaps**:
  - **if `input_list[mid] == 0`**: Swap with the element at `low` and increment both `low` and `mid`. This step ensures 0s are moved to the beginning.
  - **elif `input_list[mid] == 1`**: Simply move the `mid` pointer, leaving 1s in the middle of the array.
  - **else (`input_list[mid] == 2`)**: Swap with the element at `high` and decrement `high`. This ensures 2s are moved to the end.
