from sklearn.linear_model import LinearRegression
import PySimpleGUI as sg
from Matplotlib_Functions import set_fig, str_to_list, create_data, mse, grad, gradient_descent, create_constants
from PySimpleGUI_Functions import delete_figure_agg, draw_figure
import numpy as np

if __name__ == '__main__':

    sg.change_look_and_feel('Dark Blue')

    # define the window layout
    layout = [[sg.Text('Gradient Descent')],
              [sg.Canvas(key='-CANVAS-')],
              [sg.T('Actual X'), sg.T('Actual Y')],
              [sg.Input('0.1 1.2 2.4 3.2 4.1 5.7 6.5', key='x', size=(22, 1)),
               sg.Input('1.7 2.4 3.5 3.0 6.1 9.4 8.2', key='y', size=(22, 1))],
              [sg.T('                   '), sg.T('Predicted Y (ŷ)'), sg.Text(size=(22, 1), key='out')],
              [sg.T('Multiplier:'), sg.Input('0.01', key='multi')],
              [sg.T('Number of Iterations:'), sg.Input('1000', key='iter')],
              [sg.T('Number of Test Points:'), sg.Input('200', key='nr_thetas')],
              [sg.Button('Ok')]]

    # create the form and show it without the plot
    window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True,
                       element_justification='center', font='Helvetica 18', location=(200, 0),
                       icon='3d_graph_icon.ico')

    fig_canvas_agg = None

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
                plot_cost, plot_t0, plot_t1 = create_constants(x_5=x_5, y_5=y_5, nr_thetas=nr_thetas)
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

            plot_vals, mse_vals, thetas = gradient_descent(x_5=x_5, y_5=y_5, multiplier=multiplier, length=length)

            final_mse = mse(y_5, thetas[0] + thetas[1] * x_5)

            fig = set_fig(plot_vals=plot_vals, mse_vals=mse_vals, plot_cost=plot_cost, plot_t0=plot_t0,
                          plot_t1=plot_t1, thetas=thetas, mse=final_mse)

            fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig, fig_canvas_agg=fig_canvas_agg)

            y_hat = thetas[0] * x_5 + thetas[1]
            y_hat = np.round(y_hat, 1)
            window['out'].update(y_hat.tolist())

            window.refresh()
            print('Done!')

    window.close()