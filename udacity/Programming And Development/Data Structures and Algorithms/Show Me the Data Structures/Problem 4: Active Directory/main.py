class Group(object):
    def __init__(self, _name):
        self.name = _name
        self.groups = []
        self.users = []

    def add_group(self, group):
        self.groups.append(group)

    def add_user(self, user):
        self.users.append(user)

    def get_groups(self):
        return self.groups

    def get_users(self):
        return self.users

    def get_name(self):
        return self.name


def is_user_in_group(user, group):
    """
    Return True if user is in the group, False otherwise.

    Args:
      user(str): user name/id
      group(class:Group): group to check user membership against
    """
    if user in group.get_users():
        return True

    for subgroup in group.get_groups():
        if is_user_in_group(user, subgroup):
            return True

    return False


# Test Cases
if __name__ == "__main__":
    parent = Group("parent")
    child = Group("child")
    sub_child = Group("subchild")

    sub_child_user = "sub_child_user"
    sub_child.add_user(sub_child_user)

    child.add_group(sub_child)
    parent.add_group(child)

    print(f"Test Case 1: User is in subchild group")
    print(is_user_in_group("sub_child_user", parent))  # True

    print(f"Test Case 2: User is not in any group")
    print(is_user_in_group("random_user", parent))  # False

    print(f"Test Case 3: User is in the root group")
    parent_user = "parent_user"
    parent.add_user(parent_user)
    print(is_user_in_group("parent_user", parent))  # True

    # Edge cases
    print(f"Test Case 4: Empty user check")
    print(is_user_in_group("", parent))  # False

    print(f"Test Case 5: User in an empty group")
    empty_group = Group("empty_group")
    empty_group_user = "empty_group_user"
    empty_group.add_user(empty_group_user)
    print(is_user_in_group("empty_group_user", empty_group))  # True

    print(f"Test Case 6: Check in None group")
    try:
        print(is_user_in_group("some_user", None))  # Should handle gracefully
    except AttributeError:
        print("Caught an AttributeError as expected when checking a None group")
