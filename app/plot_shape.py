#!/usr/bin/env python3
import plotly.graph_objects as go


def plot_square(
    fig, center_x, center_y, side_length, color="lightblue", line_color="black"
):
    """
    Adds a square shape to a Plotly figure.

    Parameters:
    - fig: The figure to which the square is added
    - center_x: x-coordinate of the square's center
    - center_y: y-coordinate of the square's center
    - side_length: Length of the square's side
    - color: Fill color of the square
    - line_color: Border color of the square
    """
    half_length = side_length / 2

    # Calculate the four vertices of the square
    x0 = center_x - half_length
    y0 = center_y - half_length
    x1 = center_x + half_length
    y1 = center_y - half_length
    x2 = center_x + half_length
    y2 = center_y + half_length
    x3 = center_x - half_length
    y3 = center_y + half_length

    fig.add_shape(
        type="path",
        path=f"M {x0} {y0} L {x1} {y1} L {x2} {y2} L {x3} {y3} Z",
        fillcolor=color,
        line=dict(color=line_color),
    )


def plot_rectangle(
    fig, center_x, center_y, width, length, color="lightblue", line_color="black"
):
    """
    Adds a rectangular shape to a Plotly figure.

    Parameters:
    - fig: The figure to which the rectangle is added
    - center_x: x-coordinate of the rectangle's center
    - center_y: y-coordinate of the rectangle's center
    - length: Length of the rectangle
    - width: Width of the rectangle
    - color: Fill color of the rectangle
    - line_color: Border color of the rectangle
    """
    half_length = length / 2
    half_width = width / 2

    # Calculate the four vertices of the rectangle
    x0 = center_x - half_width
    y0 = center_y - half_length
    x1 = center_x + half_width
    y1 = center_y - half_length
    x2 = center_x + half_width
    y2 = center_y + half_length
    x3 = center_x - half_width
    y3 = center_y + half_length

    fig.add_shape(
        type="path",
        path=f"M {x0} {y0} L {x1} {y1} L {x2} {y2} L {x3} {y3} Z",
        fillcolor=color,
        line=dict(color=line_color),
    )


def rectangleShape(cl, w, h):
    x0 = cl["x"]
    y0 = cl["y"]
    shapes = [
        {
            "x": [x0 + w / 2, x0 - w / 2, x0 - w / 2, x0 + w / 2, x0 + w / 2],
            "y": [y0 + h / 2, y0 + h / 2, y0 - h / 2, y0 - h / 2, y0 + h / 2],
        }
    ]
    return shapes


def addShape(
    fig, shapes, name="Square", line_type="solid", fill_option=False, color="blue"
):
    # Add all the shapes to the figure
    for shape in shapes:
        fill_option = None if fill_option == False else "toself"
        fig.add_trace(
            go.Scatter(
                x=shape["x"],
                y=shape["y"],
                mode="lines+markers",
                fill=fill_option,
                name=name,
                line=dict(dash=line_type, color=color),
                marker=dict(color=color),
            ),
        )


def add_Hline(fig, y_value, bounds, color="RoyalBlue"):
    # Add a horizontal line at y = y_value
    fig.add_shape(
        type="line",
        x0=bounds[0],
        y0=y_value,
        x1=bounds[1],
        y1=y_value,  # Replace -10 and 10 with your x-axis bounds
        line=dict(dash="dash", color=color, width=3),
    )


def add_Vline(fig, x_value, bounds, color="Red"):
    # Add a vertical line at x = x_value
    fig.add_shape(
        type="line",
        x0=x_value,
        y0=bounds[0],
        x1=x_value,
        y1=bounds[1],  # Replace -10 and 10 with your y-axis bounds
        line=dict(dash="dash", color=color, width=3),
    )
