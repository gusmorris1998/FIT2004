from collections import deque
# Problem 1
class AdjacencyList:
    def __init__(self, num_vertices):
        """
        Function description: Constructor for adjacency list class

        Complexity:

        let n be the variable, number of vertices

        time complexity O(n)
        space comexity O(n)
        """
        self.num_vertices = num_vertices
        self.adj_list = [[] for i in range(num_vertices)]

    def add_edge(self, source, dest, weight):
        """
        Function description: Adds and edge to the adjacency list
        
        Complexity:

        O(1) time and space complexity.
        """
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
    use in the Edmand-Karp algorithm that uses both the graph and residual network to find
    any augmenting paths. 


    Complexity:

    Let V be the vertices and E be the edges in the graphs.

    Each vertex is visited once and the edges are viewed at most twice therefore time complexity
    is O(V + E)

    Graphs are stored as adjacency lists, hence space comlexity is equal to O(V + E)

    
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

# Problem 2
class CatsTrie: 
    """
    Class description: Class containing a prefix trie, using the regular 26 word english alphabet
    """
    def __init__(self, sentences):
        """
        Function Description: Constructor for CatsTrie class. The class contains a prefix trie, which
        is made with an array of arrays, where each index contains to the corresponding character for that
        index. Eg, a = 0, z = 25

        Function Approach: Initialise the trie by assigning an index for each character including the $, and
        have a 0 in the 27th index of the array which corresponds the highest order of all branches below it
        The order of a branch corresponds to the greatest occurences of a single word in that branch. For each
        sentence given O(n) that sentence is then inserted into the trie O(m)

        Complexity:
        time complexity O(N * M), where n is the longest sentence and m is the amount of sentences, as for each sentence that
        is inserted it may have to traverse down the n length branch during insertion.

        space complexity O(N * M), assuming we consider alphabet size constant as for each sentence, every m could be a
        different longest length string, meaning it takes up N * M space.
        """
        self.trie = [None] * 27 + [0]
        for sentence in sentences:
            self.insert(sentence)

    def insert(self, word):
        """
        Function Description: Takes a word as input then places that word into the prefix trie.
        Function Approach: Traverse the characters in the word then for each character find the place
        within the array for which character that associates to in the array. If the place in that array
        None then a new array needs to be created that is associated to that character. The ord() function
        is used to find the index in the array.

        Once whole word is traversed a $ is placed in the 26th index of the arrayto indicate the end of a word
        the order is incremented for that letter. The order of the entire branch is then incremented by by use
        of the update_order() method.

        Complexity: let W be the length of the word, then the whole word is traversed and the inner contents of
        the loop is O(1) so hence the time complexity of the loop is O(W)

        The update_order() function is also O(W) therefore the time complexity of the insert function is O(W)
        """
        node = self.trie
        for char in word:
            # Map character to array index (0-25)
            index = ord(char) - ord('a')
            if node[index] is None:
                node[index] = [None] * 27 + [0]
            node = node[index]
        node[26] = '$'
        node[27] += 1

        self.update_order(word, node[27])

    def update_order(self, word, order):
        """
        Function Description: Updates the order of a branch to match the maximum order word within that
        branch

        Function Approach: Given the integer value order as input its starts from the top of the trie and works
        downwards for each character checking within that branch if the given input order is greater than the order
        of that branch that is contained at index 27.

        Complexity:

        Constant work is done for each character traversed within the word, hence time complexity is O(W) where w
        is the word length.
        """
        node = self.trie
        if node[27] < order:
            node[27] = order

        for char in word:
            # Map character to array index (0-25)
            index = ord(char) - ord('a')
            node = node[index]
            if node[27] < order:
                node[27] = order

    def autoComplete(self, prompt):
        """
        Function Description: Given a prompt, autocompletes the sentence by returning the highest
        frequency sentence that has the prompt as a prefix. If there are multiple choices for the highest
        frequency then the lexicographically smallest choice is picked.

        Function Approach: The approach I used relies on the order of each branch being kept in it. The order of a
        branch corresponds to the highest frequency sentence within that branch. Firstly the prompt is check
        to exist by traversing through it. If the prompt does not finish traversing then it does not exist and
        therefore can not be autocompleted so return None.

        If the prompt is within the prefix trie it can then be autocompleted so a predeccessor string is kept that
        is added to until the autocomplete is finished. In order to autocomplete the highest order branch is followed
        down, by traversing through each array, where the first occurence that matches the order will be added to the
        predecessor string. Once the character is added, you move further down the branch and do the same.

        This terminates when the are no more branches of the max order, and hence will reach the dollar sign. At
        this point the predecessor string can be returned, which will be the autocompleted sentence

        Complexity: 
        O(X + Y) time complexity where x is the length of the prompt and Y is the length of the most frequent sentence
        begining with the prompt is. This can be seen in the first for loop which run X times with constant inner contents.
        If an autocomplete exists then it will perform a further Y iterations in the second loop. 

        Note that despite the entire alphabet being traversed in the second loop, this is O(1) as the loop is the alphabet
        length is consider constant time.
        """
        node = self.trie
        for char in prompt:
            # Map character to array index (0-25)
            index = ord(char) - ord('a')
            if node[index] is None:
                return None
            node = node[index]
        order = node[27]
        i = 0

        predecessor = prompt
        while node[i] != '$':
            if node[i] and node[i][27] == order:
                predecessor += chr(i + ord('a'))
                node = node[i]
                i = 0
            else:
                i += 1
        return predecessor

