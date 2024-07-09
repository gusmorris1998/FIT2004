from collections import deque

# Problem 1

class AdjacencyList:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adj_list = [[] for _ in range(num_vertices)]

    def add_edge(self, source, dest, weight):
        self.adj_list[source].append((dest, weight))

def maxThroughput(connections, maxIn, maxOut, origin, targets):
    """
    Function description: This function returns an integer value, that is equal to the maximal
    flow through a modified network, that is used to calculates the maxiumum posible data throughput
    from an origin its targets,

    Function approach: In order to calculate maximal flow through a network I used a variant of the
    the Ford-Fulkerson algorithm which was implemented with a bread-first search. This was done in order
    to exist within the complexity guidelines of O(|V||E|^2). Before Edmand-Karp could be run, the graph
    given had to be transformed to suitable input.

    Firstly, each vertex had a flow restriction on it, independant of its incoming and outgoing edges. In order
    to abide by the requirements, I converted each vertex (v) into 2 vertices vIn and vOut. All incoming edges
    of v is redirected to vIn and each outgoing edge of v now came from vOut. For the vertex restriction on v, an
    edges was added from vIn to vOut that was equal to the minimum of its max inflow or max outflow of v. Exceptions
    applied to the source vertex and target vertices. For the source (s) the edge from sIn to sOut was equal to vOut
    as it was only responsible for sending flow. For any target vertex (t) the edge between tIn to tOut was equal to
    the max inflow of each t as targets are only responsible for recieving flow.

    As multiple targets can not be processed by a traditional Ford-Fulkerson algorithm, a universal sink node was
    created such that an edge between each tOut was direct to the universal sink, equal to the t's vertex flow
    restriction.

    Following both of those modifications, Ford-Fulkerson could be then ran on transformed graph in order to calculate
    the maximal flow.

    Complexity: let |D| be the data centres and |C| be the communication channels which can be abstracted to graph
    vertices and edges respectively.

    So after modification |D| = 2 * |D| + 1, |C| = |C| + |D|, ignoring constants the complexity of running Edman-Karp is:

    |D|(|D| + |C|) ^2 = |D|(|D|^2 + 2|D||C|  + |C|^2)
                      = |D|^3 + 2|D|^2|C| + |D||C|^2 

    For which |D||C|^2 is faster growing than the other two, hence time complexity is |D||C|^2
    """
    
    # Transform graph
    vertices = 2 * len(maxIn) + 1
    graph = AdjacencyList(vertices)
    
    # Edges between vertices, 
    for edge in connections:
        uOut, vIn, weight = 2 * edge[0] + 1, 2 * edge[1], edge[2]
        graph.add_edge(uOut, vIn, weight)

    # Capacities of nodes
    for i in range(len(maxIn)):
        uIn, uOut = 2 * i, 2 * i + 1
        if uIn / 2 != origin:
            capacity = min(maxIn[i], maxOut[i])
        else:
            capacity = maxOut[i]
        graph.add_edge(uIn, uOut, capacity)

    # 
    for target in targets:
        targetIn, targetOut = 2 * target, 2 * target + 1
        # Adjust targets capacity to maxIn[target]
        graph.adj_list[targetIn][0] = (targetOut, maxIn[target])
        universalSink = vertices - 1
        # Weight of edge between vIn and vOut where v is the target node
        weight = graph.adj_list[targetIn][0][1]
        graph.add_edge(targetOut, universalSink, weight)

    return edmand_karp(graph, origin, universalSink)

def edmand_karp(adjacency_list, source, sink):
    """
    Function description: Variant of Ford-Fulkerson algorithm that uses breadth first search. Flow
    is initialized to 0 on all edges, BFS is then performed on the residual graph in order to find the
    augmenting path, for which the minimum capacity of the augmenting path is found where the flow and
    capacity is updated on each iteration. This is done for all augmenting paths for which the maximum
    flow can be obtained.
    
    Complexity: Let V be the vertices and E be the edges

    Each iteration of the algorithm performs a BFS to find an augmenting path. O(V + E), within these
    iterations the update of flow is done in O(E) time hence the time complexity for Edmand-Karp is 
    O(V)

    Both the residual graph and the adjacency graph is equal to O(V + E) as they both use adjacency list
    representation, hence space complexity is O(2(V+E)) = O(V + E) space complexity for edmand-karp.
    """
    num_vertices = adjacency_list.num_vertices

    # Initialize the residual graph with the capacities from the adjacency list
    residual_graph = [[0] * num_vertices for i in range(num_vertices)]
    for u in range(num_vertices):
        for v, capacity in adjacency_list.adj_list[u]:
            residual_graph[u][v] = capacity

    max_flow = 0

    # Augment the flow while there is a path from source to sink in the residual graph
    path_found = True
    while path_found:
        path_found, parent = modified_bfs(adjacency_list, source, sink, residual_graph)
        if path_found:
            # Find the minimum capacity edge along the augmenting path
            min_capacity = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                min_capacity = min(min_capacity, residual_graph[u][v])
                v = u

            # Update the residual capacities and reverse edges along the augmenting path
            v = sink
            while v != source:
                u = parent[v]
                residual_graph[u][v] -= min_capacity
                residual_graph[v][u] += min_capacity
                v = u

            # Add the minimum capacity to the max flow
            max_flow += min_capacity

    return max_flow

def modified_bfs(adjacency_graph, source, sink, residual_graph):
    """
    Function description: modified version of breadth first search algorithm to be
    use in the Edmand-Karp algorithm
    """
    num_vertices = adjacency_graph.num_vertices

    # Initialize visited and parent arrays for BFS
    visited = [False] * num_vertices
    parent = [-1] * num_vertices

    # Create a queue for BFS
    queue = deque()
    queue.append(source)
    visited[source] = True

    # BFS loop
    while queue:
        u = queue.popleft()

        # Explore the neighbors of u in the residual graph
        for v, capacity in adjacency_graph.adj_list[u]:
            if not visited[v] and residual_graph[u][v] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u

                # If we reached the sink node in BFS, there is a path from source to sink
                if v == sink:
                    return True, parent

    return False, parent





