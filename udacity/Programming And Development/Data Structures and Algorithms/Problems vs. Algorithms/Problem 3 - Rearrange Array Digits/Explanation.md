### Explanation:

#### Efficiency:
The solution's efficiency is primarily derived from its use of the Merge Sort algorithm, which is known for its time complexity of \(O(n \log n)\). This plays a critical role in sorting the input list efficiently. Given the constraints and requirements, this approach ensures scalability even with larger input sizes.

#### Breakdown of Efficiency:
- **Merge Sort (`merge_sort` function)**:
  - **Splitting**: The array is recursively split into halves until each element is isolated, achieving \(O(\log n)\) splits.
  - **Merging**: During merging, each element is compared and appended to the sorted list, leading to \(O(n)\) comparisons. Together, this results in \(O(n \log n)\).
  
#### Design Choices:
- **Choosing Merge Sort**:
  - **Efficiency**: Merge Sort is chosen for its stable performance with average and worst-case time complexity of \(O(n \log n)\), ensuring consistency.
  - **Descending Order Sorting**: The algorithm allows easy modification to sort in descending order, which is essential for maximizing the sum when forming two numbers.
  
#### Alternative Approaches Considered:
- **Bubble Sort/Insertion Sort**: These have a worst-case time complexity of \(O(n^2)\), which is inefficient for larger inputs.
- **Quick Sort**: While it has an average-case time complexity of \(O(n \log n)\), its worst-case time complexity of \(O(n^2)\) makes it less reliable without additional optimizations (such as random pivots).

#### Rearrange Digits:
- **Digit Distribution**:
  - **Alternating Assignment**: By alternately appending digits to `num_str1` and `num_str2`, we ensure the sums are maximized. This simple but effective approach ensures each number gets the highest remaining digit in every step.
  
#### Overall Complexity:
- The combined operations in the `rearrange_digits` function include sorting \(O(n \log n)\) and linear passes for alternating assignment \(O(n)\).

Combining these, the overall complexity of the function is \(O(n \log n)\), making it efficient and suitable for the given problem.

### Design Considerations:
- **Simplicity and Clarity**: The code is designed to be understandable, with clear functions encapsulating the sorting and rearranging operations.
- **Edge Cases**: Handles scenarios like empty lists and lists with zeros naturally, returning (0, 0) appropriately.
