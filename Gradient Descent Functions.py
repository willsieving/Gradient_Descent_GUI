import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm # Color map

from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib

matplotlib.use('TkAgg')
sg.change_look_and_feel('Dark Blue')

def set_fig(plot_vals, mse_vals, plot_cost, plot_t0, plot_t1, thetas, mse):
    global fig
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_title(f'Theta_0: {round(thetas[0], 5)} - Theta_1: {round(thetas[1], 5)}\nMSE:{mse[0]}')
    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')
    ax.set_zlabel('Cost - MSE')

    ax.scatter(plot_vals[:, 0], plot_vals[:, 1], mse_vals, color='black')
    ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.rainbow, alpha=0.6)


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    # This function removes the outdated graph so it can be replaced


def draw_figure(canvas, figure):
    try:
        # if a graph already exists, delete it to make room for next one
        delete_figure_agg(fig_canvas_agg)
    except NameError:
        pass
    # Draw new graph
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


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

def gradient_descent(multiplier=0.01, thetas=np.array([2.9, 2.9]), length=1000):

    # Collect Data Point for Scatter Plot
    plot_vals = thetas.reshape(1, 2)
    mse_vals = mse(y_5, thetas[0] + thetas[1] * x_5)

    for i in range(length):
        thetas = thetas - multiplier * grad(x_5, y_5, thetas)
        # if slopes are larger then subtract more from thetas
        # make bigger steps if slopes are larger

        print(thetas)

        # Append the new values to our numpy arrays
        plot_vals = np.concatenate((plot_vals, thetas.reshape(1, 2)), axis=0)
        mse_vals = np.append(mse_vals, values=mse(y_5, thetas[0] + thetas[1] * x_5))

    return plot_vals, mse_vals, thetas

def create_constants(nr_thetas=200):

    th_0 = np.linspace(start=-1, stop=3, num=nr_thetas)
    th_1 = np.linspace(start=-1, stop=3, num=nr_thetas)

    plot_t0, plot_t1 = np.meshgrid(th_0, th_1)
    plot_cost = np.zeros((nr_thetas, nr_thetas))

    for i in range(nr_thetas):
        for j in range(nr_thetas):
            y_hat = plot_t0[i][j] + plot_t1[i][j] * x_5
            plot_cost[i][j] = mse(y_5, y_hat)

    return plot_cost, plot_t0, plot_t1


# define the window layout
layout = [[sg.Text('Gradient Descent')],
          [sg.Canvas(key='-CANVAS-')],
          [sg.T('Actual X'), sg.T('Actual Y')],
          [sg.Input('0.1 1.2 2.4 3.2 4.1 5.7 6.5', key='x', size=(22, 1)), sg.Input('1.7 2.4 3.5 3.0 6.1 9.4 8.2', key='y', size=(22, 1))],
          [sg.T('Predicted Y (ŷ)'), sg.Text(size=(22, 1), key='out')],
          [sg.T('Multiplier:'), sg.Input('0.01', key='multi')],
          [sg.T('Number of Iterations:'), sg.Input('1000', key='iter')],
          [sg.T('Number of Test Points:'), sg.Input('200', key='nr_thetas')],
          [sg.Button('Ok')]]


# create the form and show it without the plot
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True,
                   element_justification='center', font='Helvetica 18', location=(200, 0))


while True:
    event, values = window.read()

    if event is None:
        break
    if event == 'Ok':
        try:
            nr_thetas = int(values['nr_thetas'])
        except:
            pass

        try:
            x_5, y_5 = create_data(x=values['x'], y=values['y'])

        except:
            x_5, y_5 = create_data()

        try:
            plot_cost, plot_t0, plot_t1 = create_constants(nr_thetas=nr_thetas)
        except:
            plot_cost, plot_t0, plot_t1 = create_constants()

        try:
            multiplier = float(values['multi'])
        except:
            multiplier = 0.01

        try:
            length = int(values['iter'])
        except:
            length = 1000


        regr = LinearRegression()
        regr.fit(x_5, y_5)

        plot_vals, mse_vals, thetas = gradient_descent(multiplier=multiplier, length=length)

        final_mse = mse(y_5, thetas[0] + thetas[1]*x_5)

        set_fig(plot_vals=plot_vals, mse_vals=mse_vals, plot_cost=plot_cost, plot_t0=plot_t0, plot_t1=plot_t1, thetas=thetas, mse=final_mse)
        fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

        y_hat = thetas[0]*x_5 + thetas[1]
        y_hat = np.round(y_hat, 1)
        window['out'].update(y_hat.tolist())

        window.refresh()
        print('Done!')

window.close()
