# Name: Gus Morris

def dijkstra(graph, start):
    """
    Function description: Dijkstra's algorithm is a shortest path algorithm that finds the shortest path between 
    a starting node and all other nodes in a weighted graph. It uses a priority queue to determine the next node 
    to visit and updates the distances to all neighboring nodes if a shorter path is found. The algorithm keeps 
    track of the previous node in the shortest path for each visited node, allowing the optimal path to be reconstructed.

    Approach: approach mainly inspired from the lecture notes.

    The time complexity of dijksta's algorithm is O(E log V) which is O(R log V) in our case.

    """
    # initialize the distances and predecessor arrays
    n = len(graph)
    distances = [float('inf')] * n
    predecessors = [None] * n
    # set the distance to the starting node to 0
    distances[start] = 0
    # initialize the heap with the starting node and its distance
    heap = [(0, start)]
    # initialize the visited list
    visited = [False] * n
    while heap:
        # pop the smallest distance node from the heap
        (dist, node) = pop_min(heap)
        if visited[node]:
            continue
        visited[node] = True
        # iterate over the node's neighbors
        for neighbor, weight in graph[node]:
            # calculate the new distance to the neighbor
            new_distance = dist + weight
            # update the distance and predecessor if it's shorter than the current one
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = node
                # add the neighbor and its distance to the heap
                push(heap, (new_distance, neighbor))
    return (distances, predecessors)

# implementation of heap operations

def push(heap, item):
    heap.append(item)
    sift_up(heap, len(heap) - 1)

def pop_min(heap):
    last = heap.pop()
    if heap:
        min_item = heap[0]
        heap[0] = last
        sift_down(heap, 0)
    else:
        min_item = last
    return min_item

def sift_up(heap, i):
    parent = (i - 1) // 2
    while i > 0 and heap[i][0] < heap[parent][0]:
        heap[i], heap[parent] = heap[parent], heap[i]
        i = parent
        parent = (i - 1) // 2

def sift_down(heap, i):
    left = 2 * i + 1
    right = 2 * i + 2
    smallest = i
    if left < len(heap) and heap[left][0] < heap[smallest][0]:
        smallest = left
    if right < len(heap) and heap[right][0] < heap[smallest][0]:
        smallest = right
    if smallest != i:
        heap[i], heap[smallest] = heap[smallest], heap[i]
        sift_down(heap, smallest)

def max_vertice(roads):
    """
    Function description: iterates through roads to find the max vertice within the array of tuples
    Approach: iterate list of all tuples and check both the start and end vertex

    Uses a for loop to iterate over all the roads, therefore O(R). Aux space complexity O(1)

    Input:
        roads: list of tuples contain vertex from, vertex to and distances.
    Output:
        max_vert: an integer representing the max vertex in the tuple.
    """
    max_vert = 0
    # Iterate over all the tuples in the list O(R)
    for i in range(len(roads)):
        # Checks both the first and second item in tuples assigning max accordingly
        if roads[i][0] > max_vert:
            max_vert = roads[i][0]
        if roads[i][1] > max_vert:
            max_vert = roads[i][1] 
    return max_vert


def create_graph(roads, max_vert, reverse = False):
    """
        Function description: The function returns an adjacency list of tuples, where the list index refers to the start
        vertex, the first element in the tuple contains the end vertex and the second element in the tuple contains the
        distance from the the the vertices

        Approach: by using the max_vert parameter the graph could be initialized such that all the vertices have a given list
        index. A parameter was used so that this would be the forward graph or the reverse graph, though the approaches to
        creating these graphs were fundamentally the same. Iterate through the roads and assign the tuples with the end
        and edge weight to the according list index depending on where the vertex is coming from.

        creating vertices O(L), creating edges O(R) both time and space.

        therefore, Space and Time complexity: O(L + R)

        Inputs:
            roads: List of tuples, vertice and edge weights
            max_vert: maximum vertices in roads, integer
            reverse: boolean that determines whether a regular or reversed graph.

    """

    # O(L) creates the 'from' vertices
    graph = [[] for i in range(0, max_vert + 1)]
    # Conditional to determine whether this will be the reverse graph
    if not reverse:
        # This creates a regular graph in which the both the direction and edge weights are regular O(R)
        for road in roads:
            start, end, weight = road[0], road[1], road[2]
            graph[start].append((end, weight))
    else:
        # The reverse case. The directions will be reversed and the toll weights will be used O(R)
        for road in roads:
            start, end, weight = road[1], road[0], road[3]
            graph[start].append((end, weight))

    return graph
        
def optimalRoute(start, end, passengers, roads):

    """
        Function description: This function returns a list containing the optimal route from a start vertex to an end vertex, 
        via the use of Djkestra's Algorithm on an adjacency lists.

        Approach: By first finding the max vertice within the roads, I built both a regular graph, and a reverse graph with
        the edges reversed. By doing doing this I could first run Djkestra on the regular graph and identify the shortest
        distances from all the vertices using the non-carpool distances. I then calculated the optimal route from the end vertex
        to the all others using the carpool times. I could then use the distances from the end to find the optimal routes from
        the end to the passengers. By combinining these results I could then find the optimal route from the start to the end.

        Let R be the roads(edges) and L be the key locations(vertices), thus: 

        max_vertice: O(L) as the roads were iterated through as well as the vertices to find the max edge.
        create_graph: O(L + R) both time and space complexity as each road is itereated through and edges and vertices
            are stored.
        djkestra: O(R log L) time complexity and O(L + R) space complexity
        finding optimal path length: O(P)
        find_optimal_route: O(L)

        Inputs
            start: a starting location (int)
            end: an  ending location (int)
            passengers: a list of locations where passengers will be (List[int])
            roads: a list of tuples containing a start, end, regualar distance, toll distance List[(start, end, regular, toll)]

        Output
            optimal_route: a list of vertices in order of the quickest route from start to end

        therefore as the functions are not dependant on each other it can be seen that the time complexity is 
        bound by O(R log L) and the space complexity is bound by O(L + R)

    """

    # Find the maximum vertice within the roads
    max_vert = max_vertice(roads)

    # Initialize forward graph
    graph = create_graph(roads, max_vert)
    # Forward Dijkestra, returns a pred
    (forward_dist, forward_pred) = dijkstra(graph, start)
    # Initialize reverse graph
    graph = create_graph(roads, max_vert, True)
    # Reverse Dijkestra
    (reverse_dist, reverse_pred) = dijkstra(graph, end)

    # Regular optimal path length
    optimal_route_length = forward_dist[end]
    taken_passenger = None

    # Determine optimal path length by summating the reverse carpool distances and regular non-carpool distances O(P)
    for passenger in passengers:
        # Conditional to check against currently held optimal_route_length.
        if forward_dist[passenger] + reverse_dist[passenger] < optimal_route_length:
            optimal_route_length = forward_dist[passenger] + reverse_dist[passenger]
            taken_passenger = passenger

    # Conditional checks whether any passenger was taken during the optimal route.
    if not taken_passenger:
        # Returns a path such that no carpool lanes are taken.
        return find_optimal_route(forward_pred, start, end, False)
    else:
        # Returns a path such that the regular route to the taken_passenger is calculated and route from the end to the passengers
            # is calculated, then the two solutions are merged to reach the final optimal path.
        return find_optimal_route(forward_pred, start, taken_passenger, False) + find_optimal_route(reverse_pred, end, taken_passenger, True)



def find_optimal_route(predecessor_array, start_node, end_node, reversed):
    """
        Function description: iterates through a predecessor array in order to return an optimal path
        Approach: begin with an array containing only the end node from this each node is iterated through until
        the start is then equal to the end node. As we are working with lists that are both regular and reversed,
        a boolean: reversed is given as a parameter, so that if it is regular it then needs to be revered in order
        to give the correct direction, but if reversed is True, then it will not return the reversed list but instead
        remove the first element so that there is no duplicate node within the path.

        in order to iterate through the pred list these take O(L) time complexity and to reverse the list is also O(L)
        therefore the time complexity of this function is O(L) and the space complexity is also O(L) as a new list of
        length L is created.

        Inputs:
            predecessor_array: an array contains the list index as a vertex and the item in the index the predecessor to
                that vertex
            start_node: int representing start node
            end_node: int representing end node
            reversed: boolean to represent whether the pred_array corresponds to a reversed pred array.

        Output:
            array containing the given path from start to end.
    """
    # Initialize the route with the end node
    route = [end_node]
    
    # Follow the predecessor array backwards from the end node to the start node
    current_node = end_node
    # O (L)
    while current_node != start_node:
        current_node = predecessor_array[current_node]
        route.append(current_node)
    
    # Reverse the route to get it in the correct order if a forward pred
    if not reversed:
        # O(L)
        route.reverse()
    else:
        # Removes first item from reverse route so no duplicate nodes
        route = route[1:]
    
    return route

def min_index(lst):
    """
        Function descriptions: returns the min index of a given list
        Approach: iterates through the list keeping track of the index

        O(n) where n is the len of the list

    """
    m = len(lst)
    min_index = 0
    # iterates over the list
    for i in range(1, m):
        if lst[i] < lst[min_index]:
            # keeps track of index
            min_index = i
    return min_index

def select_sections(op):
    """
        Function description: given a list of lists this finds the optimal selection to remove for each row in a 
        company so that the minimal total occupancy is lost.
        Approach: I used a dynamic programming approach to overide the orginal occupancy array, by using an optimal
        substructure in which the new values are equal to that of the current i, jth value + minimum of i-1th,[j-1:j+1]th
        value was used to create a memo array. Then since the given memo array contain the largest values at the nth row,
        we could then use this to back track back through the memo array, keeping track of the optimal route at each row.

        In order to first override the original array, two for loops are used traversing (1 to n) and (1 to m). Therefore
        the given complexity of this block is O(m*n), however the auxillary space is O(1) as the original array is overriden.

        After this the array is then back tracked via all n rows, but only by constant = 3 as the entire jth row is not
        needed in order to compute the optimal route from here.

        Hence time complexity is O(m*n) and space complexity is O(1)

        Input:
            op: list of lists containing the occupancy probability of each space in the office.
        Output:
            minimal_total_occupancy: the minimal total occupancy that can be effectively removed from the office
            sections_locations: the location coordinates of the sections that can be optimally removed from the office.

    """
    
    # The length by width of the array
    n, m = len(op), len(op[0])

    # Edge case where m == 0
    if m == 0:
        return [0, []]

    # Edge case where m == 1: returns the summations of all the inner arrays
    if m == 1:
        return [sum([op[i][0] for i in range(n)]), [(i, 0) for i in range(n)]]

    # O(m*n)
    # Iterates over all list O(n)
    for i in range(1, n):
        # Iterates over all inner list O(m)
        for j in range(m):
        # Forms the basis for our optimal substructure and recurrence relation. Calculates a replaces the values
        # in the op array depending by adding the i,jth value by the previous values in i-1th row and the j-1, j and j+1
        # values depending on where in the array j is currently at
            # if j on the far left, only compute i-1,jth and i-1,j+1th values
            if j == 0: op[i][j] += min(op[i - 1][j], op[i - 1][j + 1])
            # if j on the far right, only compute i-1,jth and i-1,j-1th values
            elif j == m - 1: op[i][j] += min(op[i - 1][j - 1], op[i - 1][j])
            # otherwise compute all values i-1,[j-1:j+1]th
            else: op[i][j] += min(op[i - 1][j - 1], op[i - 1][j], op[i - 1][j + 1])
    
    # Starting value for minimum_total_occupancy
    minimum_total_occupancy = op[n - 1][0]
    # Starting value for sections_location
    sections_location = [(n - 1, 0)]
    # Finds the minimum value in the last row, by iterating over the nth row. O(m)
    for j in range(1, m):
        # Changes value if there is a lower value in the nth row.
        if minimum_total_occupancy > op[n - 1][j]:
            # reassigns variables accordingly
            minimum_total_occupancy = op[n - 1][j]
            sections_location = [(n - 1, j)]

    # Iterates from the n-1th row upwards O(n). 
    for i in range(n - 2, -1, -1):
        # assigns j to the first tuple in sections_location
        j = sections_location[0][1]
        # determines the coordinates
        if j == 0: sections_location = [(i, (j) + (min_index(op[i][j:j + 2])))] + sections_location
        elif j == m - 1: sections_location = [(i, (j - 1) + (min_index(op[i][j-1:j+1])))] + sections_location
        else: sections_location = [(i, (j - 1) + (min_index(op[i][j-1:j+2])))] + sections_location

    return [minimum_total_occupancy, sections_location]

        


