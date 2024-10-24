import numpy as np
import matplotlib.pyplot as plt

def universal_Compare_Data_Plot(data1,data2,dt,axlabels = [],datatitles = [],Line_width=1.5,Black_and_white = False,Save_image_as = "", **kwargs):
    plt.close('all')
    if (len(data1.shape) > 2) or (len(data2.shape) > 2):
        print("Data is not in 2D array form")
        return
    if data1.shape != data2.shape:
        print("Two data not same shape")
        return
    if len(data1.shape) ==1:
        data1 = np.array([data1])
        data2 = np.array([data2])
    if data1.shape[0] > data1.shape[1]:
        data1 = data1.T
        data2 = data2.T            #The data should be "longer" than the dimensions. This helps with universality
    if len(axlabels) == 0:
        if data1.shape[0] > 3:
            axlabels = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10"]
        else: axlabels = ["X", "Y", "Z"]
    if len(datatitles) ==0:
        datatitles = ["Data 1", "Data 2"]

    if Black_and_white:
        blackNwhite_args1 = {"linestyle":'dashed',"color":'black'}
        blackNwhite_args2 = {"color": 'black',"alpha":0.4}
    else:
        blackNwhite_args1 = {}
        blackNwhite_args2 = {}

    for i in range(data1.shape[0]):     #Basic 2d plot by dimensions
        axs = plt.figure(figsize=(3,3)).add_subplot()
        time = np.array(range(data1.shape[1]))*dt
        axs.plot(time,data1[i],**blackNwhite_args1,linewidth=Line_width, label=datatitles[0])
        axs.plot(time,data2[i],**blackNwhite_args2,linewidth=Line_width, label=datatitles[1])
        axs.set_xlabel("Time")
        axs.set_ylabel(axlabels[i])
        #plt.legend()
        if Save_image_as != "": plt.savefig("Images\\" + Save_image_as +  "data_comparison_dimension_" + str(i) + ".pdf",bbox_inches='tight')
        else: plt.show()



    if data1.shape[0] == 1:          #1D data
        print()
    elif data1.shape[0] == 2:        #2D data
        axs = plt.figure(figsize=(4,4)).add_subplot()
        axs.plot(*data1,**blackNwhite_args1,linewidth=Line_width, label=datatitles[0])
        axs.plot(*data2,**blackNwhite_args2,linewidth=Line_width, label=datatitles[1])
        axs.set_xlabel(axlabels[0])
        axs.set_ylabel(axlabels[1])
        plt.legend()
        if Save_image_as != "": plt.savefig("Images\\" + Save_image_as + "data_comparison_2d_" + ".pdf",bbox_inches='tight')
        else: plt.show()
    elif data1.shape[0] == 3:        #3D data

        ax = plt.figure(figsize=(4,4)).add_subplot(projection='3d')
        ax.plot(*data1,**blackNwhite_args1, linewidth=Line_width, label=datatitles[0])
        ax.set_xlabel(axlabels[0])
        ax.set_ylabel(axlabels[1])
        ax.set_zlabel(axlabels[2])
        ax.plot(*data2,**blackNwhite_args2, linewidth=Line_width, label=datatitles[1])
        plt.legend()
        if Save_image_as != "": plt.savefig("Images\\" + Save_image_as + "data_comparison_3d_" + ".pdf",bbox_inches='tight')
        else: plt.show()
    else:                           #Multidim data
        print()
    return

def multiple_histogram_W_out(multiple_W_out, in_labels, out_labels, Cutoff_small_weights = 0., Figheight = -1., Figwidth = 8., Black_and_white = False, Save_image_as="", Always_show_inputs=False,**kwargs):
    plt.close('all')
    #Prework on data
    for w_out_num in range(multiple_W_out.shape[0]-1):
        if multiple_W_out[w_out_num].shape != multiple_W_out[w_out_num+1].shape: return
    #delete small weighted rows from everything
    if Always_show_inputs:  i = multiple_W_out[0].shape[1]
    else:   i = 0

    while i < multiple_W_out.shape[2]:
        delete = True
        for j in range(multiple_W_out[0].shape[0]):
            for w_out_num in range(multiple_W_out.shape[0]):
                if abs(multiple_W_out[w_out_num][j, i]) > Cutoff_small_weights:
                    delete = False
                if delete: break
        if delete:

            multiple_W_out = np.delete(multiple_W_out, i, axis=2)
            out_labels = np.delete(out_labels, i, axis=0)
        else:
            i += 1

    combinators = multiple_W_out[0].shape[1]  # it's the input  dimension of W_out (larger)
    dimensions = multiple_W_out[0].shape[0]  # it's the output dimension of W_out (smaller)
    num_w_outs = multiple_W_out.shape[0]
    if combinators != len(out_labels): out_labels = np.full(1000, " ")
    if dimensions != len(in_labels): in_labels = np.full(1000, " ")

    #Plotting starts here
    y_pos = np.array(range(0,combinators*num_w_outs,num_w_outs))
    # make the coloring and hatching
    if Black_and_white:
        base_colors = plt.cm.Greys(np.linspace(1,0.4, num_w_outs))
        defaults = ["","","//","\\","||","-","+","x","o","O",".","*"]
        base_hatchings = np.full(multiple_W_out.shape[0],{})
        for i in range(base_hatchings.shape[0]):
            base_hatchings[i] = ({"hatch" : defaults[i % len(defaults)], "edgecolor" : "black"})
    else:
        base_colors = plt.cm.rainbow(np.linspace(0, 1, num_w_outs))
        base_hatchings = np.full(num_w_outs, ({"hatch" : ""}))

    hatching = np.full(num_w_outs,0,dtype=object)
    for w_out_num in range(num_w_outs):
        hatching[w_out_num] = base_hatchings[w_out_num]


    colors = np.full(num_w_outs,0.,dtype=object)
    for w_out_num in range(num_w_outs):
        colors[w_out_num] = base_colors[w_out_num]

    #make one plot for not normed
    normed = False
    fig, axs = plt.subplots(1, dimensions)
    if Figheight == -1.:
        Figheight = combinators*num_w_outs/2.
    fig.set_figheight(Figheight)
    fig.set_figwidth(Figwidth)

    for dimension in range(dimensions):
        for w_out_num in range(num_w_outs):
            axs[dimension].barh(0.1*num_w_outs + y_pos+w_out_num*0.8, multiple_W_out[w_out_num][dimension, :], color=colors[w_out_num],**(hatching[w_out_num]))

        axs[dimension].set_yticks(y_pos+0.1*num_w_outs)
        if dimension == 0:
            axs[dimension].set_yticklabels(out_labels)
        else: axs[dimension].set_yticklabels([])

        axs[dimension].set_ylim(combinators*num_w_outs - 0.5, -.5)
        if normed :
            xlims = (-0.,0.)
            for w_out_num in range(num_w_outs):
                if np.min(multiple_W_out[w_out_num][dimension, :])<xlims[0]:
                    xlims = (np.min(multiple_W_out[w_out_num][dimension, :]),xlims[1])
                if np.max(multiple_W_out[w_out_num][dimension, :]) > xlims[1]:
                    xlims = (xlims[0],np.max(multiple_W_out[w_out_num][dimension, :]))
            dist = xlims[1]-xlims[0]
            xlims = (xlims[0]-dist*0.05,xlims[1]+dist*0.05)

            axs[dimension].set_xlim(*xlims)
        axs[dimension].set_xlabel("Pred. " + str(in_labels[dimension]))
        axs[dimension].grid(axis='x')
        #add the horizontal lines
        for row in range(combinators):
            axs[dimension].axhline(y=row*num_w_outs-0.4, color='black', linestyle='-')
    if Save_image_as != "": plt.savefig("Images\\" + Save_image_as +  "histogr_wouts_" + str(num_w_outs) + "_norm_" + str(normed) + ".pdf",bbox_inches='tight')
    else: plt.show()

    return