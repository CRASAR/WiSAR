import matplotlib.pyplot as plt
from matplotlib import patches

def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height

def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

def compare_poi_for_image(image,
    predicted_poi,
    actual_poi,
    out_path,
    figsize=(40, 20),
    suptitle="",
    suptitle_fontsize=12,
    point_size=140,
    draw_bboxes_fn=draw_pascal_voc_bboxes,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    ppoi_x = [e[0] for e in predicted_poi]
    ppoi_y = [e[1] for e in predicted_poi]

    apoi_x = [e[0] for e in actual_poi]
    apoi_y = [e[1] for e in actual_poi]

    ax1.scatter(ppoi_x, ppoi_y, edgecolors='white', c='red', s=point_size)
    ax2.scatter(apoi_x, apoi_y, edgecolors='white', c='red', s=point_size)

    fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    plt.savefig(out_path)
    plt.close()


def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    actual_bboxes,
    out_path,
    figsize=(40, 20),
    suptitle="",
    suptitle_fontsize=12,
    draw_bboxes_fn=draw_pascal_voc_bboxes
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes)
    draw_bboxes_fn(ax2, actual_bboxes)

    fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    plt.savefig(out_path)
    plt.close()

def show_image(
    image, path, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.savefig(path)
    plt.close()