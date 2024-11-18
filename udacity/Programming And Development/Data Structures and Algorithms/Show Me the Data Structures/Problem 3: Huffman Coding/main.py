import heapq
import sys
from collections import defaultdict, namedtuple


class HuffmanNode:
    def __init__(self, char=None, freq=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def calculate_frequency(data):
    freq = defaultdict(int)
    for char in data:
        freq[char] += 1
    return freq


def build_huffman_tree(freq):
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heapq.heappop(heap)


def build_codes_tree(node, current_code='', codes={}):
    if node is None:
        return

    if node.char is not None:
        codes[node.char] = current_code

    build_codes_tree(node.left, current_code + '0', codes)
    build_codes_tree(node.right, current_code + '1', codes)
    return codes


def huffman_encoding(data):
    if not data:
        return "", None

    freq = calculate_frequency(data)
    huffman_tree = build_huffman_tree(freq)
    huffman_codes = build_codes_tree(huffman_tree)

    encoded_data = ''.join([huffman_codes[char] for char in data])
    return encoded_data, huffman_tree


def huffman_decoding(encoded_data, huffman_tree):
    if not encoded_data or huffman_tree is None:
        return ""

    decoded_data = []
    current_node = huffman_tree
    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.char is not None:
            decoded_data.append(current_node.char)
            current_node = huffman_tree

    return ''.join(decoded_data)


if __name__ == "__main__":
    codes = {}

    a_great_sentence = "The bird is the word"

    print(f"The size of the data is: {sys.getsizeof(a_great_sentence)}")
    print(f"The content of the data is: {a_great_sentence}")

    encoded_data, tree = huffman_encoding(a_great_sentence)

    print(f"The size of the encoded data is: {sys.getsizeof(int(encoded_data, base=2))}")
    print(f"The content of the encoded data is: {encoded_data}")

    decoded_data = huffman_decoding(encoded_data, tree)

    print(f"The size of the decoded data is: {sys.getsizeof(decoded_data)}")
    print(f"The content of the encoded data is: {decoded_data}")


    # Additional test cases
    def run_tests():
        test_cases = [
            "",  # Empty string
            "aaaaaa",  # Single repeated character
            "abcabcabcabc",  # Multiple repeated characters
            "a quick brown foxes jumps over the lazy dog",  # Random sentence
            "abcd" * 1000  # Large input
        ]

        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test}")
            encoded_data, tree = huffman_encoding(test)
            if encoded_data and tree:
                decoded_data = huffman_decoding(encoded_data, tree)
                print(f"Original Size: {sys.getsizeof(test)}")
                print(f"Encoded Size: {sys.getsizeof(int(encoded_data, 2))}")
                print(f"Decoded Size: {sys.getsizeof(decoded_data)}")
                print(f"Encoding: {encoded_data}")
                print(f"Decoding: {decoded_data}")
                assert test == decoded_data, "Decoding failed!"
            else:
                print("(Empty or invalid input)")


    run_tests()
