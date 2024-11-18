def merge_sort(arr):
    """
    Function to perform merge sort on the array in descending order.
    Args:
    arr (List[int]): The list of integers to sort.

    Returns:
    List[int]: Sorted list in descending order.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    # Merge the two halves in descending order
    return merge(left_half, right_half)


def merge(left, right):
    """
    Merge two sorted lists into a single sorted list in descending order.
    Args:
    left (List[int]): The left half list.
    right (List[int]): The right half list.

    Returns:
    List[int]: Merged and sorted list in descending order.
    """
    sorted_list = []
    while left and right:
        if left[0] > right[0]:
            sorted_list.append(left.pop(0))
        else:
            sorted_list.append(right.pop(0))

    # Append any remaining elements
    sorted_list.extend(left)
    sorted_list.extend(right)

    return sorted_list


def rearrange_digits(input_list):
    """
    Rearranges the elements of the given array to form two numbers such that their sum is maximized.
    The numbers formed have a number of digits differing by no more than one.

    Args:
    input_list (List[int]): A list of integers

    Returns:
    Tuple[int, int]: A tuple containing two integers
    """
    if not input_list:
        return 0, 0

    # Step 1: Sort the input list in descending order using merge sort
    sorted_list = merge_sort(input_list)

    # Step 2: Distribute the digits into two numbers
    num_str1, num_str2 = "", ""

    for i, digit in enumerate(sorted_list):
        # Alternately append to num_str1 and num_str2
        if i % 2 == 0:
            num_str1 += str(digit)
        else:
            num_str2 += str(digit)

    # Step 3: Form numbers from the digit strings
    num1 = int(num_str1) if num_str1 else 0
    num2 = int(num_str2) if num_str2 else 0

    return (num1, num2)


def test_function(test_case):
    """
    Testing function to validate the output of rearrange_digits function.
    Args:
    test_case (List): Test case containing input list and expected output tuple.

    Returns:
    None
    """
    output = rearrange_digits(test_case[0])
    solution = test_case[1]

    # Convert both results to strings and sort to compare the digits
    if sorted(str(output[0]) + str(output[1])) == sorted(str(solution[0]) + str(solution[1])):
        print("Pass")
    else:
        print("Fail")


# Test cases to verify the solution
test_function([[1, 2, 3, 4, 5], [542, 31]])
test_function([[4, 6, 2, 5, 9, 8], [964, 852]])
test_function([[1, 2, 3, 4, 5, 6], [642, 531]])
test_function([[9, 1, 2, 3], [93, 21]])
test_function([[0, 0, 0, 0, 0], [0, 0]])
