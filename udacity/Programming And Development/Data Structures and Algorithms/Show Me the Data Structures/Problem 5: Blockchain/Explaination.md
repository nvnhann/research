### Explanation

#### Efficiency

**Time Complexity:**
- The time complexity for creating a new block is \(O(1)\) since it involves generating a hash and appending the block to the list.
- The time complexity for computing the hash of a block is \(O(1)\), as it involves a fixed number of operations.

**Space Complexity:**
- The space complexity for storing the chain of blocks is \(O(n)\), where \(n\) is the number of blocks in the blockchain.

#### Code Design

- **Block Class:**
  - Contains properties like index, timestamp, data, previous block's hash, and the current block's hash.
  - The `hash_block` method generates the SHA-256 hash of the block's data for integrity and link to the previous block.
  
- **Blockchain Class:**
  - Manages the chain of blocks and provides methods to add new blocks and retrieve the latest block.
  - Creates the genesis block (the first block in the chain) when the blockchain is instantiated.

#### Readability

- The code is structured logically and follows Python naming conventions.
- Each class and method has a clear purpose, making the code easy to understand.
- The example usage demonstrates how to create a blockchain, add blocks, and print the blockchain.
