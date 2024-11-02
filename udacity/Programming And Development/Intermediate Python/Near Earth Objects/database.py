"""A dataset collections of near-Earth objects and their close approaches.

A `NEODatabase` holds an interconnected data set of NEOs and close approaches.
It provides methods to fetch an NEO by primary designation or by name, as well
as a method to query the set of close approaches that match a collection of
user-specified criteria.

"""


class NEODatabase:
    """A database of near-Earth objects and their close approaches.

    A `NEODatabase` contains a collection of NEOs and a collection of close
    approaches. It additionally maintains a few auxiliary data structures to
    help fetch NEOs by primary designation or by name and to help speed up
    querying for close approaches that match criteria.
    """

    def __init__(self, neos, approaches):
        """Create a new `NEODatabase`.

        This constructor assumes that the collections of NEOs
        and close approaches haven't yet been linked - that is, the
        `.approaches` attribute of each `NearEarthObject` resolves to an empty
        collection, and the `.neo` attribute of each `CloseApproach` is None.

        :param neos: A collection of `NearEarthObject`s.
        :param approaches: A collection of `CloseApproach`es.
        """
        self._neos = neos
        self._approaches = approaches
        self.neos_dict = dict()
        self.neos_dict_name = dict()
        for neo in self._neos:
            # add NEO object to neos_dict with key = designation
            self.neos_dict[neo.designation] = neo
            # if NEO has a name, add NEO object to neos_dict_name with key = name
            if neo.name:
                self.neos_dict_name[neo.name] = neo
        for approach in self._approaches:
            # if NEO has a name, add NEO object to neos_dict_name with key = name
            approach.neo = self.neos_dict[approach._designation]
            # append approach object to NEO's list of approaches
            self.neos_dict[approach._designation].approaches.append(approach)

    def get_neo_by_designation(self, designation):
        """Find and return an NEO by its primary designation.

        If no match is found, return `None` instead.

        Each NEO in the data set has a unique primary designation, as a string.

        The matching is exact - check for spelling and capitalization if no
        match is found.

        :param designation: The primary designation of the NEO to search for.
        :return: The `NearEarthObject` with the desired primary designation, or `None`.
        """
        return self.neos_dict.get(designation, None)

    def get_neo_by_name(self, _name):
        """Find and return an NEO by its name.

        If no match is found, return `None` instead.

        Not every NEO in the data set has a name. No NEOs are associated with
        the empty string nor with the `None` singleton.

        The matching is exact - check for spelling and capitalization if no
        match is found.

        :param name: The name, as a string, of the NEO to search for.
        :return: The `NearEarthObject` with the desired name, or `None`.
        """
        return self.neos_dict_name.get(_name, None)

    def query(self, filters=()):
        """Query close approaches to generate those that match a collection of filters.

        This generates a stream of `CloseApproach` objects that match all of the
        provided filters.

        If no arguments are provided, generate all known close approaches.

        The `CloseApproach` objects are generated in internal order, which isn't
        guaranteed to be sorted meaningfully, although is often sorted by time.

        :param filters: A collection of filters capturing user-specified criteria.
        :return: A stream of matching `CloseApproach` objects.
        """
        if not filters:
            yield from self._approaches
            return
        for approach in self._approaches:
            # Initialize a flag as True. If the flag remains True, it means that the approach matches all the filters.
            matches_all_filters = True
            # Iterate through all the filters and check if the approach matches each one of them.
            for filter_func in filters:
                if not filter_func(approach):
                    # If the approach doesn't match a filter, set the flag to False and break out of the loop.
                    matches_all_filters = False
                    break
            # If the flag is still True after the loop, yield the approach.
            if matches_all_filters:
                yield approach
