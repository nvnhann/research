class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return str(self.value)


class LinkedList:
    def __init__(self):
        self.head = None

    def __str__(self):
        cur_head = self.head
        out_string = ""
        while cur_head:
            out_string += str(cur_head.value) + " -> "
            cur_head = cur_head.next
        return out_string

    def append(self, value):
        if self.head is None:
            self.head = Node(value)
            return

        node = self.head
        while node.next:
            node = node.next

        node.next = Node(value)

    def size(self):
        size = 0
        node = self.head
        while node:
            size += 1
            node = node.next

        return size


def union(llist_1, llist_2):
    # Tạo set lưu trữ các giá trị duy nhất
    elements = set()

    # Thêm các giá trị từ llist_1 vào set
    current = llist_1.head
    while current:
        elements.add(current.value)
        current = current.next

    # Thêm các giá trị từ llist_2 vào set
    current = llist_2.head
    while current:
        elements.add(current.value)
        current = current.next

    # Tạo danh sách kết quả từ set
    result_llist = LinkedList()
    for element in elements:
        result_llist.append(element)

    return result_llist


def intersection(llist_1, llist_2):
    # Tạo set lưu trữ các giá trị từ llist_1
    elements_1 = set()
    current = llist_1.head
    while current:
        elements_1.add(current.value)
        current = current.next

    # Tạo set lưu trữ các giá trị từ llist_2
    elements_2 = set()
    current = llist_2.head
    while current:
        elements_2.add(current.value)
        current = current.next

    # Tìm giao của hai set
    intersection_elements = elements_1.intersection(elements_2)

    # Tạo danh sách kết quả từ giao của hai set
    result_llist = LinkedList()
    for element in intersection_elements:
        result_llist.append(element)

    return result_llist


if __name__ == "__main__":
    # Test case 1
    linked_list_1 = LinkedList()
    linked_list_2 = LinkedList()

    element_1 = [3, 2, 4, 35, 6, 65, 6, 4, 3, 21]
    element_2 = [6, 32, 4, 9, 6, 1, 11, 21, 1]

    for i in element_1:
        linked_list_1.append(i)

    for i in element_2:
        linked_list_2.append(i)

    print("Union of linked_list_1 and linked_list_2:")
    print(union(linked_list_1, linked_list_2))
    print("Intersection of linked_list_1 and linked_list_2:")
    print(intersection(linked_list_1, linked_list_2))

    # Test case 2
    linked_list_3 = LinkedList()
    linked_list_4 = LinkedList()

    element_1 = [3, 2, 4, 35, 6, 65, 6, 4, 3, 23]
    element_2 = [1, 7, 8, 9, 11, 21, 1]

    for i in element_1:
        linked_list_3.append(i)

    for i in element_2:
        linked_list_4.append(i)

    print("Union of linked_list_3 and linked_list_4:")
    print(union(linked_list_3, linked_list_4))
    print("Intersection of linked_list_3 and linked_list_4:")
    print(intersection(linked_list_3, linked_list_4))

    # Add your own test cases: include at least three test cases
    # and two of them must include edge cases, such as null, empty or very large values
    ## Test Case 1
    linked_list_5 = LinkedList()
    linked_list_6 = LinkedList()

    element_1 = [i for i in range(1000)]
    element_2 = [i for i in range(500, 1500)]

    for i in element_1:
        linked_list_5.append(i)

    for i in element_2:
        linked_list_6.append(i)

    print("Union of linked_list_5 and linked_list_6:")
    print(union(linked_list_5, linked_list_6))
    print("Intersection of linked_list_5 and linked_list_6:")
    print(intersection(linked_list_5, linked_list_6))

    ## Test Case 2: Empty linked list
    linked_list_7 = LinkedList()
    linked_list_8 = LinkedList()

    print("Union of empty linked_list_7 and linked_list_8:")
    print(union(linked_list_7, linked_list_8))
    print("Intersection of empty linked_list_7 and linked_list_8:")
    print(intersection(linked_list_7, linked_list_8))

    ## Test Case 3: Single element linked list
    linked_list_9 = LinkedList()
    linked_list_10 = LinkedList()

    linked_list_9.append(1)
    linked_list_10.append(1)

    print("Union of linked_list_9 and linked_list_10 with single element:")
    print(union(linked_list_9, linked_list_10))
    print("Intersection of linked_list_9 and linked_list_10 with single element:")
    print(intersection(linked_list_9, linked_list_10))
