class RouteTrieNode:
    def __init__(self):
        """
        Initialize the node with children as before, plus a handler.
        """
        self.children = {}
        self.handler = None

class RouteTrie:
    def __init__(self, root_handler):
        """
        Initialize the trie with a root node and a root handler.
        """
        self.root = RouteTrieNode()
        self.root.handler = root_handler

    def insert(self, path_parts, handler):
        """
        Recursively add nodes. Assign handler to the leaf node of this path.
        """
        current_node = self.root
        for part in path_parts:
            if part not in current_node.children:
                current_node.children[part] = RouteTrieNode()
            current_node = current_node.children[part]
        current_node.handler = handler

    def find(self, path_parts):
        """
        Navigate the Trie to find a match for the path.
        Return the handler for a match, or None for no match.
        """
        current_node = self.root
        for part in path_parts:
            if part not in current_node.children:
                return None
            current_node = current_node.children[part]
        return current_node.handler


class Router:
    def __init__(self, root_handler, not_found_handler=None):
        """
        Initialize with a RouteTrie for holding our routes.
        Add a handler for 404 page not found responses as well.
        """
        self.route_trie = RouteTrie(root_handler)
        self.not_found_handler = not_found_handler

    def add_handler(self, path, handler):
        """
        Add a handler for a path.
        Split path and pass parts to the RouteTrie.
        """
        path_parts = self.split_path(path)
        self.route_trie.insert(path_parts, handler)

    def lookup(self, path):
        """
        Lookup path and return the associated handler.
        Return the "not found" handler if not found.
        Handle trailing slashes.
        """
        path_parts = self.split_path(path)

        if not path_parts:
            return self.route_trie.root.handler

        handler = self.route_trie.find(path_parts)
        if handler:
            return handler
        return self.not_found_handler

    @staticmethod
    def split_path(path):
        """
        Split the path into parts for the add_handler and lookup functions.
        """
        return [part for part in path.strip("/").split("/") if part]

# Create the router and add a route
router = Router("root handler", "not found handler")
router.add_handler("/home/about", "about handler")  # add a route

# Some lookups with the expected output
print(router.lookup("/"))  # should print 'root handler'
print(router.lookup("/home"))  # should print 'not found handler'
print(router.lookup("/home/about"))  # should print 'about handler'
print(router.lookup("/home/about/"))  # should print 'about handler'
print(router.lookup("/home/about/me"))  # should print 'not found handler'

# Additional test cases
router.add_handler("/home/contact", "contact handler")
print(router.lookup("/home/contact"))  # should print 'contact handler'
print(router.lookup("/home/contact/"))  # should print 'contact handler'
print(router.lookup("/home/contact/me"))  # should print 'not found handler'

router.add_handler("/home/contact/me", "contact me handler")
print(router.lookup("/home/contact/me"))  # should print 'contact me handler'
print(router.lookup("/home/contact/me/"))  # should print 'contact me handler'

# Edge cases
print(router.lookup(""))  # should print 'root handler'
print(router.lookup("/random"))  # should print 'not found handler'
