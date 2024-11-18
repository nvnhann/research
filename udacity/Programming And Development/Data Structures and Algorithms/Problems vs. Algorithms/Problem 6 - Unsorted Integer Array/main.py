def get_min_max(ints):
    """
    Return a tuple(min, max) out of list of unsorted integers.

    Args:
       ints (list): list of integers containing one or more integers

    Returns:
       tuple: (min, max)
    """
    if not ints:
        return None  # Handle the empty list case, though the problem assumes at least one integer

    # Initialize min and max values with the first element of the list
    min_val = max_val = ints[0]

    # Iterate through the list starting from the second element
    for num in ints[1:]:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num

    return (min_val, max_val)


# Example Test Case of Ten Integers
import random

l = [i for i in range(0, 10)]  # a list containing 0 - 9
random.shuffle(l)

print("Pass" if ((0, 9) == get_min_max(l)) else "Fail")

# Additional Test Cases
print(get_min_max([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]))  # Should print (1, 9)
print(get_min_max([10]))  # Should print (10, 10)
print(get_min_max([37, 12, 25, 89, 47]))  # Should print (12, 89)
print(get_min_max([-1, -2, 0, 1, 2]))  # Should print (-2, 2)
print(get_min_max([119, 29, 34, 87, 654, 2, 19]))  # Should print (2, 654)
