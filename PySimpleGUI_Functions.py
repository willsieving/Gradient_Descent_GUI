import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def set_progress_window(length):
    progress_layout = [[sg.Text('      \n                                               ', key='_UPDATETEXT_')],
                       # Character count must be large for large update text later on
                       [sg.ProgressBar(length, orientation='h', size=(20, 20), key='progressbar')],
                       [sg.Cancel(button_text='Stop')]]

    window = sg.Window('Running Gradient Descent...', progress_layout, finalize=True, icon='3d_graph_icon.ico')
    progress_bar = window['progressbar']
    return window, progress_bar


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    print('------DELETED PREVIOUS GRAPH------')
    # This function removes the outdated graph so it can be replaced


def draw_figure(canvas, figure, fig_canvas_agg):
    try:
        # if a graph already exists, delete it to make room for next one
        delete_figure_agg(fig_canvas_agg)
    except AttributeError:
        pass
    # Draw new graph
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
