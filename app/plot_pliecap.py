import plotly.graph_objects as go

from plot_shape import rectangleShape, addShape, add_Hline, add_Vline


def plot_pilecap(
    w,
    h,
    B,
    L,
    d,
    mx,
    my,
    vx,
    vy,
    origin,
    piles_number,
    pileSize,
    planedPile,
    actualPile,
):
    fig = go.Figure()
    fig.update_layout(showlegend=False, title="Foundation Layout")

    # Column and pilecap shapes
    column_shapes = rectangleShape(origin, w, h)
    footing_shapes = rectangleShape(origin, B, L)
    addShape(fig, column_shapes, color="#b3bcc9")
    addShape(fig, footing_shapes, color="#b3bcc9")

    # Planed piles shapes
    for coord, label in zip(planedPile, piles_number):
        CL = {"x": coord["x"], "y": coord["y"]}
        planedPileShape = rectangleShape(CL, pileSize, pileSize)
        addShape(fig, planedPileShape, color="#1a66a1")

        # Annotations
        fig.add_annotation(
            text=str(label),
            x=coord["x"],
            y=coord["y"],
            showarrow=False,
            font=dict(color="red", size=11),
        )

    # Actual piles shapes
    for coord, label in zip(actualPile, piles_number):
        CL = {"x": coord["x"], "y": coord["y"]}
        actualPileShape = rectangleShape(CL, pileSize, pileSize)
        addShape(fig, actualPileShape, color="#eb9234")

    # Critical line of M
    add_Vline(fig, mx, [-L / 2, L / 2], color="green")
    add_Hline(fig, my, [-B / 2, B / 2], color="green")

    # Critical line of V
    add_Vline(fig, vx, [-L / 2, L / 2], color="blue")
    add_Hline(fig, vy, [-B / 2, B / 2], color="blue")

    # Ctrical bound of punching
    punching_shapes = rectangleShape(origin, w + d * 1e-2, h + d * 1e-2)
    addShape(fig, punching_shapes, line_type="dash", fill_option=False, color="#eb9234")

    fig.show()
