import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm # Color map
import PySimpleGUI as sg
import matplotlib

from PySimpleGUI_Functions import set_progress_window

matplotlib.use('TkAgg')



def set_fig(plot_vals, mse_vals, plot_cost, plot_t0, plot_t1, thetas, mse):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_title(f'Theta_0: {round(thetas[0], 5)} - Theta_1: {round(thetas[1], 5)}\nMSE:{mse[0]}')
    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')
    ax.set_zlabel('Cost - MSE')

    ax.scatter(plot_vals[:, 0], plot_vals[:, 1], mse_vals, color='black')
    ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.rainbow, alpha=0.6)

    return fig


def str_to_list(string):
    num_list = string.split(' ')
    num_list = [float(i) for i in num_list]
    return num_list


def create_data(x=np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]),
                y=np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2])):
    if type(x) is str and type(y) is str:
        x_list = str_to_list(x)
        y_list = str_to_list(y)
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
    # Input X and Y strings to get data
    x_5 = x.reshape(7, 1)
    y_5 = y.reshape(7, 1)
    return x_5, y_5


# Implementation of Mean Squared Error
def mse(y, y_hat):
    total = 0
    for values in zip(y, y_hat):
        total += (values[0] - values[1])**2
    return total / y.size


# Inputs are the x values, the y values, and an array of theta values
# array will have theta0 @ index 0 and theta1 @ index 1
# these two values are what will be used to modify in our gradient descent function
def grad(x, y, thetas):
    n = y.size

    theta0_slope = (-2 / n) * sum(y - thetas[0] - thetas[1] * x)
    theta1_slope = (-2 / n) * sum((y - thetas[0] - thetas[1] * x) * x)

    # Combine the slopes into an array (3 different way to do so)
    # return np.array((theta_slope[0], theta1_slope[0]))
    # return np.append(arr=theta0_slope, values=theta1_slope)
    return np.concatenate((theta0_slope, theta1_slope), axis=0)

def gradient_descent(x_5, y_5, multiplier=0.01, thetas=np.array([2.9, 2.9]), length=1000):

    # Collect Data Point for Scatter Plot
    plot_vals = thetas.reshape(1, 2)
    mse_vals = mse(y_5, thetas[0] + thetas[1] * x_5)

    progress_window, progress_bar = set_progress_window(length=length)


    for i in range(length):
        thetas = thetas - multiplier * grad(x_5, y_5, thetas)
        # if slopes are larger then subtract more from thetas
        # make bigger steps if slopes are larger

        progress_event, progress_values = progress_window.read(timeout=10)
        if progress_event == 'Stop' or progress_event == None:
            break

        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.UpdateBar(i + 1)

        update_str = f'Calculating Thetas...\nIteration: {i}/{length}'
        progress_window['_UPDATETEXT_'].update(update_str)
        # !!!!----- IMPORTANT -----!!!!
        # Your update text must be =< characters than your original text

        progress_window['_UPDATETHETAS_'].update(f'Theta_0/Y-Intercept: {thetas[0]}\nTheta_1/Slope:{thetas[1]}')




        # Append the new values to our numpy arrays
        plot_vals = np.concatenate((plot_vals, thetas.reshape(1, 2)), axis=0)
        mse_vals = np.append(mse_vals, values=mse(y_5, thetas[0] + thetas[1] * x_5))

    # Don't forget to close progress window
    progress_window.close()
    return plot_vals, mse_vals, thetas


def create_constants(x_5, y_5, nr_thetas=200):

    th_0 = np.linspace(start=-1, stop=3, num=nr_thetas)
    th_1 = np.linspace(start=-1, stop=3, num=nr_thetas)

    plot_t0, plot_t1 = np.meshgrid(th_0, th_1)
    plot_cost = np.zeros((nr_thetas, nr_thetas))

    for i in range(nr_thetas):
        for j in range(nr_thetas):
            y_hat = plot_t0[i][j] + plot_t1[i][j] * x_5
            plot_cost[i][j] = mse(y_5, y_hat)

    return plot_cost, plot_t0, plot_t1


