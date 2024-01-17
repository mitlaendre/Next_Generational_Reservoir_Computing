import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def Array_Combination_to_tuple(array = np.array([],dtype=object),combination = np.array([])):
    tuple = ()
    for i in range(array.shape[0]):
        tuple = tuple + (array[i][combination[i]],)
    return tuple

def Array_to_tuple(array = np.array([],dtype=object)):
    tuple = ()
    for i in range(array.shape[0]):
        tuple = tuple + (array[i],)
    return tuple

def error_func_mse(x, y):
    return np.average(np.power(sum(np.square(x - y)), 0.5))
def array_min_finder(input_array = np.array([0],dtype=object),maxthreads = 1): #output is a tuple with first element is the location vector (as an np.array), second is the minimum value
    if (input_array.ndim == 1):
        return np.array([np.array([np.argmin(input_array)]),np.amin(input_array)],dtype=object)
    else:
        this_dimensions_size = input_array.shape[0]
        sub_solutions = Parallel(n_jobs=min(input_array.size,maxthreads))(delayed(array_min_finder)(input_array[i],maxthreads = max(1,(maxthreads//input_array.size))) for i in range(this_dimensions_size))

        locations = np.full(this_dimensions_size,0,dtype=object)
        minimums = np.full(this_dimensions_size,0,dtype=float)
        for i in range(this_dimensions_size):
            locations[i] = sub_solutions[i][0]
            minimums[i] = sub_solutions[i][1]

        return (np.append(np.array([np.argmin(minimums)]),locations[np.argmin(minimums)]),minimums[np.argmin(minimums)])

def multidimensional_array_special_averaging(array = np.array([],dtype=object), array_IS_this_dimension_needs_averaging = np.array([],dtype=bool)):
    #the averaged dimensions of the array will still be there, but it will be 1 long
    dimensions_to_Average = ()
    for i in range(array_IS_this_dimension_needs_averaging.shape[0]):
        if(array_IS_this_dimension_needs_averaging[i] == True):
            dimensions_to_Average = dimensions_to_Average + (i,)

    for i in dimensions_to_Average:
        array = array.mean(axis = i,keepdims=True)

    return array

def plot_W_out(arr,row_labels,col_labels):
    fig, ax = plt.subplots()

    # Create the heatmap
    ax.imshow(np.abs(arr), cmap='hot')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, arr[i, j],ha="center", va="center", color="w")

    ax.set_title("W_out matrix")
    fig.tight_layout()
    plt.show()
    return

def histogram_W_out(W_out,labels):
    y_pos = np.arange(W_out.shape[1])
    colorx = ['b', 'b']
    colory = ['b', 'b']
    colorz = ['b', 'b']
    for ix in range(100):
        colorx += ['b']
        colory += ['b']
        colorz += ['b']

    fig1a, axs1a = plt.subplots(1, 3)
    fig1a.set_figheight(7.)
    fig1a.set_figwidth(6.)

    axs1a[0].barh(y_pos, W_out[0, :], color=colorx)
    axs1a[0].set_yticks(y_pos)
    #axs1a[0].set_yticklabels(labels)
    axs1a[0].set_ylim(26.5 + 1, -.5)
    axs1a[0].set_xlabel('$[W_{out}]_x$')
    axs1a[0].grid()

    axs1a[1].barh(y_pos, W_out[1, :], color=colory)
    axs1a[1].set_yticks(y_pos)
    axs1a[1].axes.set_yticklabels([])
    axs1a[1].set_ylim(26.5 + 1, -.5)
    axs1a[1].set_xlabel('$[W_{out}]_y$')
    axs1a[1].grid()

    axs1a[2].barh(y_pos, W_out[2, :], color=colorz)
    axs1a[2].set_yticks(y_pos)
    axs1a[2].axes.set_yticklabels([])  # ,rotation='vertical')
    axs1a[2].set_ylim(26.5 + 1, -.5)
    # axs1a[2].set_xlim(-.09,.1)
    # axs1a[2].set_xticks([-.08,0.,.08])
    axs1a[2].set_xlabel('$[W_{out}]_z$')
    axs1a[2].grid()

    plt.show()

    ##### zoom in ####

    fig1b, axs1b = plt.subplots(1, 3)
    fig1b.set_figheight(7.)
    fig1b.set_figwidth(6.)

    axs1b[0].barh(y_pos, W_out[0, :], color=colorx)
    axs1b[0].set_yticks(y_pos)
    #axs1b[0].set_yticklabels(labels)
    axs1b[0].set_ylim(26.5 + 1, -.5)
    axs1b[0].set_xlim(-.2, .2)
    axs1b[0].set_xticks([-0.1, 0., .1])
    axs1b[0].set_xlabel('$[W_{out}]_x$')
    axs1b[0].grid()

    axs1b[1].barh(y_pos, W_out[1, :], color=colory)
    axs1b[1].set_yticks(y_pos)
    #axs1b[1].axes.set_yticklabels([])
    axs1b[1].set_ylim(26.5 + 1, -.5)
    axs1b[1].set_xlim(-.3, .3)
    axs1b[1].set_xticks([-0.2, 0., .2])
    axs1b[1].set_xlabel('$[W_{out}]_y$')
    axs1b[1].grid()

    axs1b[2].barh(y_pos, W_out[2, :], color=colorz)
    axs1b[2].set_yticks(y_pos)
    #axs1b[2].axes.set_yticklabels([])  # ,rotation='vertical')
    axs1b[2].set_ylim(26.5 + 1, -.5)
    axs1b[2].set_xlim(-.07, .07)
    axs1b[2].set_xticks([-0.05, 0., .05])
    axs1b[2].set_xlabel('$[W_{out}]_z$')
    axs1b[2].grid()

    plt.show()
    return
