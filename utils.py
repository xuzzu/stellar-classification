from imports import *
from params import *

def visualize_2D_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=0)
    # X_tsne = tsne.fit_transform(X_small)
    X_tsne = tsne.fit_transform(features)

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labels, opacity=0.6, labels={'color': 'Class'})

    fig.update_layout(
        title="t-SNE Visualization of Data",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        coloraxis_colorbar_title="Class",
        height=1000,
        width=1000
    )

    fig.write_html(f"{plots_path}/t-SNE-2d-norm.html")

def visualize_3D_tsne(features, labels):
    tsne = TSNE(n_components=3, random_state=0)
    tsne_features = tsne.fit_transform(features)

    # make 3D plot in plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=tsne_features[:, 0],
        y=tsne_features[:, 1],
        z=tsne_features[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=labels,
            colorscale='Viridis',
            opacity=0.5
        )
    )])

    fig.update_layout(
        title="t-SNE Visualization of Data",
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3"
        )
    )

    fig.write_html(f"{plots_path}/t-SNE-3d-nn-features.html")


def tsne_3D_animation(features, labels, frames=180, rotation_angle=3):
    tsne = TSNE(n_components=3, random_state=0)
    X_tsne = tsne.fit_transform(features)

    # Create the initial 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        z=X_tsne[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=labels,
            colorscale='Viridis',
            opacity=0.5
        )
    )])

    # Set the layout for the plot
    fig.update_layout(
        title="t-SNE Visualization of Data",
        scene=dict(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            zaxis_title="Feature 3",
            xaxis=dict(range=[-35, 35]),  # Set X axis range
            yaxis=dict(range=[-35, 35]),  # Set Y axis range
            aspectmode='cube',  # Set aspect ratio to 'cube'
            aspectratio=dict(x=1, y=1, z=1)  # Set aspect ratio to 1:1:1
        ),
        height=1000,
        width=1000
    )

    # Create rotation frames
    n_frames = frames
    rotation_angle = rotation_angle
    frame_duration = 50

    frames = []
    for i in range(n_frames):
        frame = go.Frame(
            name=f'frame{i}',
            data=[go.Scatter3d(
                x=X_tsne[:, 0] * np.cos(np.deg2rad(rotation_angle * i)) - X_tsne[:, 1] * np.sin(np.deg2rad(rotation_angle * i)),
                y=X_tsne[:, 0] * np.sin(np.deg2rad(rotation_angle * i)) + X_tsne[:, 1] * np.cos(np.deg2rad(rotation_angle * i)),
                z=X_tsne[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=labels,
                    colorscale='Viridis',
                    opacity=0.6
                )
            )],
            layout=go.Layout(scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=2)
            )),
            traces=[0]
        )
        frames.append(frame)

    # Add frames to the figure's animation
    fig.frames = frames

    # Create animation
    animation = dict(
        frame=dict(duration=frame_duration, redraw=True),
        fromcurrent=True,
        mode='immediate'
    )

    # Configure the figure to play the animation automatically
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, animation]
                    )
                ],
                showactive=False,
                direction='left',
                x=0.1,
                y=0
            )
        ]
    )

    # Show the figure
    fig.write_html(f"{plots_path}/t-SNE-3d-nn-features-animation.html")