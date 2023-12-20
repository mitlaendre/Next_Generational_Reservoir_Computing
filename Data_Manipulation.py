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

def plot_3dData_3dPlot(x):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*x.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()

def plot_3dData_2dPlot(x):
    plt.plot(np.transpose(x)[0],"r-",legend = "x")
    plt.plot(np.transpose(x)[1],"g-",legend = "y")
    plt.plot(np.transpose(x)[2],"b-",legend = "z")
    plt.show()

    x1 = np.full((x.shape[0],2),0)
    for i in range(x1.shape[0]):
        x1[i,0] = x[i,0]+x[i,1]
        x1[i,1] = x[i,2]
    plt.plot(np.transpose(x1)[0],np.transpose(x1)[1])
    plt.show()

    return

def compare_3dData_3dPlot(ground_truth, prediction):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*ground_truth.T, lw=0.5, label="ground truth")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.plot(*prediction.T, lw=0.5, label="prediction")

    plt.legend()
    plt.show()
    return

def compare_3dData_2dPlot(ground_truth, prediction):
    plt.plot(np.transpose(ground_truth)[0], "r-", label="x")
    plt.plot(np.transpose(ground_truth)[1], "g-", label="y")
    plt.plot(np.transpose(ground_truth)[2], "b-", label="z")
    plt.plot(np.transpose(prediction)[0], "r--", label="Prediction x")
    plt.plot(np.transpose(prediction)[1], "g--", label="Prediciton y")
    plt.plot(np.transpose(prediction)[2], "b--", label="Prediciton z")
    plt.show()
    return

def plot_errors_surface(input_errors = np.array([]),Reservoir_sizes = np.array([]),Leaking_Rates = np.array([]),Spectral_Radiuses = np.array([])):
    for i in range(input_errors.shape[0]):
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(Leaking_Rates, Spectral_Radiuses)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.plot_surface(X, Y, input_errors[i])
        ha.set_xlabel('$Leaking Rate$')
        ha.set_ylabel('$Spectral Radius$')
        ha.set_zlabel(r'$Average error$')
        plt.show()

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
    axs1a[0].set_yticklabels(labels)
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
    axs1b[0].set_yticklabels(labels)
    axs1b[0].set_ylim(26.5 + 1, -.5)
    axs1b[0].set_xlim(-.2, .2)
    axs1b[0].set_xticks([-0.1, 0., .1])
    axs1b[0].set_xlabel('$[W_{out}]_x$')
    axs1b[0].grid()

    axs1b[1].barh(y_pos, W_out[1, :], color=colory)
    axs1b[1].set_yticks(y_pos)
    axs1b[1].axes.set_yticklabels([])
    axs1b[1].set_ylim(26.5 + 1, -.5)
    axs1b[1].set_xlim(-.3, .3)
    axs1b[1].set_xticks([-0.2, 0., .2])
    axs1b[1].set_xlabel('$[W_{out}]_y$')
    axs1b[1].grid()

    axs1b[2].barh(y_pos, W_out[2, :], color=colorz)
    axs1b[2].set_yticks(y_pos)
    axs1b[2].axes.set_yticklabels([])  # ,rotation='vertical')
    axs1b[2].set_ylim(26.5 + 1, -.5)
    axs1b[2].set_xlim(-.07, .07)
    axs1b[2].set_xticks([-0.05, 0., .05])
    axs1b[2].set_xlabel('$[W_{out}]_z$')
    axs1b[2].grid()

    plt.show()
    return
