import math
from heapq import heappop, heappush


def shortest_path(M, start, goal):
    """
    Find the shortest path from the start to the goal using the A* search algorithm.
    
    Args:
    - M: graph data structure with intersections and roads
    - start: the starting node
    - goal: the goal node
    
    Returns:
    - list: A list of nodes representing the shortest path from start to goal.
    """
    # Priority queue for the open set (frontier)
    open_set = [(0, start)]

    # Maps to track the best path cost to each node and the actual path
    came_from = {}
    g_score = {node: float('inf') for node in M.intersections}
    g_score[start] = 0
    f_score = {node: float('inf') for node in M.intersections}
    f_score[start] = heuristic(M.intersections[start], M.intersections[goal])

    # Set to keep track of visited nodes
    closed_set = set()

    while open_set:
        # Get the node with the lowest f_score
        _, current = heappop(open_set)

        # If we reach the goal, reconstruct the path
        if current == goal:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbor in M.roads[current]:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + distance(M.intersections[current], M.intersections[neighbor])

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(M.intersections[neighbor], M.intersections[goal])

                if neighbor not in [i[1] for i in open_set]:
                    heappush(open_set, (f_score[neighbor], neighbor))

    # If we reach here, no path was found
    print("No path found")
    return []


def heuristic(a, b):
    """
    Heuristic function to estimate the distance between two intersections.
    
    Args:
    - a, b: tuples representing the coordinates of the intersections
    
    Returns:
    - float: the estimated distance
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distance(a, b):
    """
    Calculate the actual distance between two intersections.
    
    Args:
    - a, b: tuples representing the coordinates of the intersections
    
    Returns:
    - float: the actual distance
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def reconstruct_path(came_from, current):
    """
    Reconstruct the path from the start to the goal.
    
    Args:
    - came_from: dictionary containing the best paths to each node
    - current: the current node (goal)
    
    Returns:
    - list: A list of nodes representing the path from start to goal.
    """
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]
