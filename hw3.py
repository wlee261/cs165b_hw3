#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#TODO: Read the input file and store it in the data structure
def read_data(path):
    """
    Read the input file and store it in data_set.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        path: path to the dataset

    Returns:
        data_set: a list of data points, each data point is itself a list of features:
            [
                [x_1, ..., x_n],
                ...
                [x_1, ..., x_n]
            ]
    """

    data_set = []
    traininglist = []
    training_file = open(path, 'r')
    for x in training_file:
        traininglist.append(x)
    training_file.close()
    for i in range(len(traininglist)):
        data_set.append([])
        line_in_traininglist = traininglist[i].split(",")
        for x in range(len(line_in_traininglist)):
            data_set[i].append(float(line_in_traininglist[x]))
    
    return data_set


# TODO: Select k points randomly from your data set as starting centers.
def init_centers_random(data_set, k):
    """
    Initialize centers by selecting k random data points in the data_set.
    
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: a list of data points, where each data point is a list of features.
        k: the number of mean/clusters.

    Returns:
        centers: a list of k elements: centers initialized using random k data points in your data_set.
                 Each center is a list of numerical values. i.e., 'vals' of a data point.
    """
    centers = []
    list_of_int = []
    while (len(list_of_int) < k):
        r = random.randint(0, len(data_set)-1)
        if r not in list_of_int:
            list_of_int.append(r)
    
    for i in range(len(list_of_int)):
        centers.append(data_set[list_of_int[i]])

    return centers


# TODO: compute the euclidean distance from a data point to the center of a cluster
def dist(vals, center):
    """
    Helper function: compute the euclidean distance from a data point to the center of a cluster

    Args:
        vals: a list of numbers (i.e. 'vals' of a data_point)
        center: a list of numbers, the center of a cluster.

    Returns:
         d: the euclidean distance from a data point to the center of a cluster
    """
    distance = 0
    #print(vals)
    #print(center)
    for i in range(len(vals)):
        distance = (vals[i] - center[i])**2 + distance
        
    return math.sqrt(distance)



# TODO: return the index of the nearest cluster
def get_nearest_center(vals, centers):
    """
    Assign a data point to the cluster associated with the nearest of the k center points.
    Return the index of the assigned cluster.

    Args:
        vals: a list of numbers (i.e. 'vals' of a data point)
        centers: a list of center points.

    Returns:
        c_idx: a number, the index of the center of the nearest cluster, to which the given data point is assigned to
    """

    min = 100000
    c_idx = 0
    for i in range(len(centers)):
        if dist(vals, centers[i]) < min:
            min = dist(vals, centers[i])
            c_idx = i

    return c_idx

# TODO: compute element-wise addition of two vectors.
def vect_add(x, y):
    """
    Helper function for recalculate_centers: compute the element-wise addition of two lists.
    Args:
        x: a list of numerical values
        y: a list of numerical values

    Returns:
        s: a list: result of element-wise addition of x and y.
    """

    s = []
    for i in range(len(x)):
        s.append(x[i] + y[i])

    return s


# TODO: averaging n vectors.
def vect_avg(s, n):
    """
    Helper function for recalculate_centers: Averaging n lists.
    Args:
        s: a list of numerical values: the element-wise addition over n lists.
        n: a number, number of lists

    Returns:
        s: a list of numerical values: the averaging result of n lists.
    """
    avg = []

    for i in range(len(s)):
        avg.append(s[i]/n)
    #print avg
    return avg

# TODO: return the updated centers.
def recalculate_centers(clusters):
    """
    Re-calculate the centers as the mean vector of each cluster.
    Args:
         clusters: a list of clusters. Each cluster is a list of data_points assigned to that cluster.

    Returns:
        centers: a list of new centers as the mean vector of each cluster.

    """
    
    centers = []
    for x in range(len(clusters)):
        count = 0
        sum = []
        for y in range(len(clusters[x][0])):
            sum.append(0)
        for i in range(len(clusters[x])):
            sum = vect_add(sum, clusters[x][i])
            count = count + 1
        centers.append(vect_avg(sum, count))
    return centers


# TODO: run kmean algorithm on data set until convergence or iteration limit.
def train_kmean(data_set, centers, iter_limit):
    """
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: a list of data points, where each data point is a list of features.
        centers: a list of initial centers.
        iter_limit: a number, iteration limit

    Returns:
        centers: a list of updates centers/mean vectors.
        clusters: a list of clusters. Each cluster is a list of data points.
        num_iterations: a number, num of iteration when converged.
    """
    list_nearest = []
    new_centers = []
    num_iterations = 0
    if(num_iterations > iter_limit):
        print("could not converge within iteration limit")
        return [], [], 0
    for i in range(len(data_set)):
        list_nearest.append(get_nearest_center(data_set[i], centers))

    cluster = []
    cluster = make_clusters(list_nearest, data_set, centers)
    new_centers = recalculate_centers(cluster)
    num_iterations = num_iterations + 1
    if(new_centers != centers):
        return train_kmean(data_set, new_centers, iter_limit)
    

    return new_centers, cluster, num_iterations


def make_clusters(list_nearest, data_set, centers):
    mcluster = []
    for i in range(len(centers)):
        mcluster.append([])
    
    for x in range(len(list_nearest)):
        mcluster[list_nearest[x]].append(data_set[x])
        
    return mcluster
# TODO: helper function: compute within group sum of squares
def within_cluster_ss(cluster, center):
    """
    For each cluster, compute the sum of squares of euclidean distance
    from each data point in the cluster to the empirical mean of this cluster.
    Please note that the euclidean distance is squared in this function.

    Args:
        cluster: a list of data points.
        center: the center for the given cluster.

    Returns:
        ss: a number, the within cluster sum of squares.
    """

    ss = 0
    for i in range(len(cluster)):
        ss = dist(cluster[i], center)**2 + ss
    
    return ss



# TODO: compute sum of within group sum of squares
def sum_of_within_cluster_ss(clusters, centers):
    """
    For total of k clusters, compute the sum of all k within_group_ss(cluster).

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        clusters: a list of clusters.
        centers: a list of centers of the given clusters.

    Returns:
        sss: a number, the sum of within cluster sum of squares for all clusters.
    """

    sss = 0
    for i in range(len(clusters)):
        sss = within_cluster_ss(clusters[i], centers[i]) + sss

    return sss

def main():
   
    data = []
    centers = []
    final = []
    xaxis = []
    yaxis = []
    maincluster = []
    iterations = 0
    data = read_data('wine.txt')
    
    
    for i in range(2,11):
        xaxis.append(i)
        centers = init_centers_random(data, i)
        final, maincluster, iterations = train_kmean(data, centers, 100)
        yaxis.append(sum_of_within_cluster_ss(maincluster, final))

    plt.plot(xaxis, yaxis)
    plt.savefig('plot.png')

if __name__ == "__main__":
	main()
