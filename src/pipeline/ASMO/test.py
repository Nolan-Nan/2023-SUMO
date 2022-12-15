from collections import Counter

# mydic = {1:{"lord bingham":["lord steyn"], "lord hope": ["lord bingham", None], "lord steyn" : ["lord bingham", None], "lord x": ["lord bingham", None]}}
#
# def to_matrix(map, case):
#     names = list(map[case].keys())
#     size = len(names)
#     matrix = [[0 for x in range(size)] for y in range(size)]
#     for key, value in map[case].items():
#         j_from = names.index(key)
#         for v in value:
#             if v is not None:
#                 j_to = names.index(v)
#                 matrix[j_from][j_to] = 1
#         print_m(matrix)
#
# def from_matrix(matrix, map, case):
#     pass
#
# def print_m(matrix):
#     for row in matrix:
#         pritem = ""
#         for item in row:
#             pritem += str(item)
#         print(pritem)
#     print("\n")
#
#
# make_matrix(mydic, 1)

class Graph:

    def __init__(self, vertices):
        self.V = vertices

    # A utility function to print the solution
    def printSolution(self, reach):
        print ("Following matrix transitive closure of the given graph ")
        for i in range(self.V):
            items = ""
            for j in range(self.V):
                items += str(reach[i][j])
            print(items)

    def warshall(self, a):
        assert (len(row) == len(a) for row in a)
        n = len(a)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    a[i][j] = a[i][j] or (a[i][k] and a[k][j])

        self.printSolution(a)
        return a


    # Prints transitive closure of graph[][] using Floyd Warshall algorithm
    def transitiveClosure(self,graph):
        '''reach[][] will be the output matrix that will finally
        have reachability values.
        Initialize the solution matrix same as input graph matrix'''
        reach =[i[:] for i in graph]
        '''Add all vertices one by one to the set of intermediate
        vertices.
         ---> Before start of a iteration, we have reachability value
         for all pairs of vertices such that the reachability values
          consider only the vertices in set
        {0, 1, 2, .. k-1} as intermediate vertices.
          ----> After the end of an iteration, vertex no. k is
         added to the set of intermediate vertices and the
        set becomes {0, 1, 2, .. k}'''
        for k in range(self.V):

            # Pick all vertices as source one by one
            for i in range(self.V):

                # Pick all vertices as destination for the
                # above picked source
                for j in range(self.V):

                    # If vertex k is on a path from i to j,
                       # then make sure that the value of reach[i][j] is 1
                    reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

        self.printSolution(reach)
        return reach

mydic = {1:{"lord bingham":["lord steyn"], "lord hope": [None], "lord steyn" : ["lord bingham"], "lord x": ["lord bingham", None]}}

def to_matrix(map, case):
    names = list(map[case].keys())
    size = len(names)
    matrix = [[0 for x in range(size)] for y in range(size)]
    for key, value in map[case].items():
        j_from = names.index(key)
        for v in value:
            if v is not None:
                j_to = names.index(v)
                matrix[j_from][j_to] = 1

    return matrix

def from_matrix(matrix, map, case):
    names = list(map[case].keys())
    for i in range(len(matrix)):
        agreed_names = []
        for j in range(len(matrix)):
            if matrix[i][j] == 1:
                agreed_names.append(names[j])
        print(names[i], agreed_names)

g = Graph(4)

graph = to_matrix(mydic, 1)
g.printSolution(graph)
#Print the solution
reach = g.transitiveClosure(graph)
from_matrix(reach, mydic, 1)

#This code is contributed by Neelam Yadav

#
# def rule_two(map, case):
#     """
#     If mj agrees with another judge. That judge is also part of MJ.
#     """
#     for k, v in map[case].items():
#         print("judge", k, v)
#         if v != [None]:
#             keys = list(v)
#             if k in keys: keys.remove(k)
#             transitive(keys, map, case, k)
#
# def transitive(keys, map, case, judge):
#     for key, value in map[case].items():
#         if value != [None]:
#             for v in value:
#                 if v == judge:
#                     original = list(map[case][key])
#                     new = list(set(original + keys))
#                     map[case][key] = new
#                     break
#
# def rule_self(map, case):
#     for k, v in map[case].items():
#         if v == [None]:
#             map[case][k] = [k]
#
# def rule_one(map, citations, case):
#     """
#     Attributes MJ to a most cited judge, unless there are more judges equally cited.
#     """
#
#     min_agreement = int(len(map[case].keys())/2) # min agreement is one below the majority of judges ie. for 5 it's 2 for 7 it's 3 for 6 it's 3
#     if min_agreement == 1:
#         min_agreement = 2
#     max = 0
#     mj = "NAN"
#
#     for k,v in citations.items():
#         if k != "NAN":
#             if v == max and mj != "NAN": # Two judges equally cited means NAN is mj
#                 mj = "NAN"
#             elif v == max and mj == "NAN": # Judge cited equally to NAN, judge is mj
#                 mj = k
#             if v > max and v >= min_agreement: # Basic rule, max cited judge is mj
#                 mj = k
#                 max = v
#
#     return mj
#
# def count_citations(map, case):
#     """
#     Counts the judges fully agreed with by the citing judge.
#     Returns counter object.
#     """
#     cited = []
#     judges = map[case].keys()
#     for judge in judges:
#         names = map[case][judge]
#         if None in names: names.remove(None)
#         names = sorted(names) # Keeps ordering of names consistent
#         if names:
#             names = ", ".join(names)
#             if judge == names:
#                 names = "NAN"
#         else: names = "NAN"
#         cited.append(names)
#
#     citations = Counter(cited)
#
#     return citations
#
# rule_two(mydic, 1)
# rule_self(mydic, 1)
# citations = count_citations(mydic, 1)
# mj = rule_one(mydic, citations, 1)
# print(mj)
#
# for k, v in mydic[1].items():
#     print(k, v)
