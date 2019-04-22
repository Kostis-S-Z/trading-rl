from bokeh.plotting import figure, output_file, show, save
import json


def plot_actions(folder, memory, long_actions, short_actions, title='agent_actions', trades=None, save_only=True):
    """
    Plot the financial positions of the agent in respect to the time series of the exchange rate.
    """
    output_file(folder + '/' + title + '.html')

    p = figure(plot_width=1000, plot_height=600)

    data_len = len(memory)

    x_axis = [x[0] for x in memory]

    y_axis = [x[1] for x in memory]

    x_axis_l = [x[0] for x in long_actions]
    y_axis_l = [y[1] for y in long_actions]
    x_axis_s = [x[0] for x in short_actions]
    y_axis_s = [y[1] for y in short_actions]

    p.line(x_axis, y_axis, line_width=2)

    p.scatter(x_axis_l, y_axis_l, marker="triangle",
              line_color="#6666ee", fill_color="#00FF00", fill_alpha=0.5, size=16)

    p.scatter(x_axis_s, y_axis_s, marker="inverted_triangle",
              line_color="#ee6666", fill_color="#FF0000", fill_alpha=0.5, size=16)

    if trades is not None:
        # Plot the time steps where a trade was made as well (long<->short)
        x_axis_tr = [x[0] for x in trades]
        y_axis_tr = [y[1] for y in trades]
        p.scatter(x_axis_tr, y_axis_tr, marker="diamond",
                  line_color="#FFA500", fill_color="#FFFF00", fill_alpha=0.5, size=16)

    if save_only:
        save(p)
    else:
        show(p)


def plot_trail(folder, memory, title='trail_history', save_only=False):
    """
    Plot the trail of the agent.
    """
    output_file(folder + '/' + title + '.html')

    p = figure(plot_width=1000, plot_height=600)

    x_axis = [x[0] for x in memory]

    data_trail = [x[1] for x in memory]
    agent_trail = [x[3] for x in memory]

    p.multi_line([x_axis, x_axis], [data_trail, agent_trail], color=["blue", "red"], line_width=2)

    if save_only:
        save(p)
    else:
        show(p)


def plot_q_values(folder, q_values, title='q_values_train', save_only=True):
    """
    Plot the q values history of the agent
    """
    output_file(folder + '/' + title + '.html')
    with open(folder + title + '.json', 'w') as f:
        json.dump(list(q_values), f)

    p = figure(plot_width=800, plot_height=600)

    x_axis = range(len(q_values))

    y1_axis = [i[0] for i in q_values]  # neutral (blue)
    y2_axis = [i[1] for i in q_values]  # buy/long (green)
    y3_axis = [i[2] for i in q_values]  # sell/short (red)

    p.multi_line([x_axis, x_axis, x_axis], [y1_axis, y2_axis, y3_axis],
                 color=["blue", "green", "red"], line_width=2)
    if save_only:
        save(p)
    else:
        show(p)


def plot_train_rewards(folder, rewards, title='train_rewards'):
    output_file(folder + '/' + title + '.html')

    p = figure(plot_width=1000, plot_height=600)

    x_axis = range(len(rewards))

    y_axis = rewards

    p.line(x_axis, y_axis, line_width=2)

    show(p)
