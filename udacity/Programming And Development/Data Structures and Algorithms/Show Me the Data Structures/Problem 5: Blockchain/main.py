import hashlib
import time


def calc_hash(input_str):
    """
    Compute the SHA-256 hash for the input string.
    """
    sha = hashlib.sha256()
    hash_str = input_str.encode('utf-8')
    sha.update(hash_str)
    return sha.hexdigest()


class Block:
    def __init__(self, timestamp, data, previous_hash=''):
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calc_hash()

    def calc_hash(self):
        """
        Compute the SHA-256 hash for the block.
        """
        input_str = f"{self.timestamp}{self.data}{self.previous_hash}"
        return calc_hash(input_str)

    def __str__(self):
        """
        Return the string representation of the block.
        """
        return f"Block - Timestamp: {self.timestamp}, Data: {self.data}, Hash: {self.hash}, Previous Hash: {self.previous_hash}"


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        """
        Create the genesis block (first block) in the blockchain.
        """
        return Block(time.gmtime(0), "Genesis Block", "0")

    def get_latest_block(self):
        """
        Get the latest block in the blockchain.
        """
        return self.chain[-1]

    def add_block(self, new_block):
        """
        Add a new block to the blockchain.
        """
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calc_hash()
        self.chain.append(new_block)

    def __str__(self):
        """
        Return the string representation of the entire blockchain.
        """
        chain_str = ""
        for block in self.chain:
            chain_str += str(block) + "\n"
        return chain_str


if __name__ == "__main__":
    # Create a new Blockchain instance
    blockchain = Blockchain()

    # Test Case 1: Add normal blocks to the blockchain
    blockchain.add_block(Block(time.gmtime(), "Block 1 Data"))
    blockchain.add_block(Block(time.gmtime(), "Block 2 Data"))

    print("Blockchain after adding two blocks:")
    print(blockchain)

    # Test Case 2: Add block with empty data
    blockchain.add_block(Block(time.gmtime(), ""))

    print("Blockchain after adding a block with empty data:")
    print(blockchain)

    # Test Case 3: Add block with large data
    large_data = "A" * 10000
    blockchain.add_block(Block(time.gmtime(), large_data))

    print("Blockchain after adding a block with large data:")
    print(blockchain)

    # Edge Case 1: Add block with None timestamp
    try:
        blockchain.add_block(Block(None, "Block with None timestamp"))
    except Exception as e:
        print("Caught an exception when adding a block with None timestamp:", e)

    # Edge Case 2: Add block with None data
    blockchain.add_block(Block(time.gmtime(), None))

    print("Blockchain after adding a block with None data:")
    print(blockchain)
