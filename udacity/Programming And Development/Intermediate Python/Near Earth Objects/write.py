"""Write a stream of close approaches to CSV or to JSON.

This module exports two functions: `write_to_csv` and `write_to_json`, each of
which accept an `results` stream of close approaches and a path to which to
write the data.

These functions are invoked by the main module with the output of the `limit`
function and the filename supplied by the user at the command line. The file's
extension determines which of these functions is used.

You'll edit this file in Part 4.
"""
import csv
import json


def write_to_csv(results, filename):
    """Write an iterable of `CloseApproach` objects to a CSV file.

    The precise output specification is in `README.md`. Roughly, each output row
    corresponds to the information in a single close approach from the `results`
    stream and its associated near-Earth object.

    :param results: An iterable of `CloseApproach` objects.
    :param filename: A Path-like object pointing to where the data should be saved.
    """
    headers = ['datetime_utc', 'distance_au',
               'velocity_km_s', 'designation', 'name',
               'diameter_km', 'potentially_hazardous']

    with open(filename, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        for close_approach in results:
            row = {'datetime_utc': close_approach.time_str,
                   'distance_au': close_approach.distance,
                   'velocity_km_s': close_approach.velocity,
                   'designation': close_approach.neo.designation,
                   'name': close_approach.neo.name,
                   'diameter_km': close_approach.neo.diameter,
                   'potentially_hazardous': close_approach.neo.hazardous}
            writer.writerow(row)


def write_to_json(results, filename):
    """Write an iterable of `CloseApproach` objects to a JSON file.

    The precise output specification is in `README.md`. Roughly, the output is a
    list containing dictionaries, each mapping `CloseApproach` attributes to
    their values and the 'neo' key mapping to a dictionary of the associated
    NEO's attributes.

    :param results: An iterable of `CloseApproach` objects.
    :param filename: A Path-like object pointing where data should be saved.
    """
    list1 = [{'datetime_utc': c.time_str,
              'distance_au': c.distance,
              'velocity_km_s': c.velocity,
              'neo': {'designation': c.neo.designation,
                      'name': c.neo.name,
                      'diameter_km': c.neo.diameter,
                      'potentially_hazardous': c.neo.hazardous}}
             for c in results]

    with open(filename, 'w') as outfile:
        json.dump(list1, outfile, indent=2)
