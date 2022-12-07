# ==========================================================================================
# THIS CODE IS FROM
# https://github.com/microsoft/CameraTraps/blob/main/visualization/visualization_utils.py
# AND
# https://github.com/microsoft/CameraTraps/blob/main/data_management/annotations/annotation_constants.py
# ==========================================================================================
import numpy as np
from PIL import ImageDraw, ImageFont

NUM_DETECTOR_CATEGORIES = 3  # this is for choosing colors, so ignoring the "empty" class


TEXTALIGN_LEFT = 0
TEXTALIGN_RIGHT = 1


COLORS = [
    "Beige",
    "Fuchsia",
    "Pink",
    "Aqua",
    "Lime",
    "Fuchsia",
]


def render_detection_bounding_boxes(
    detections,
    image,
    label_map=None,
    classification_label_map=None,
    confidence_threshold=0.8,
    thickness=4,
    expansion=0,
    classification_confidence_threshold=0.3,
    max_classifications=3,
    colormap=COLORS,
    textalign=TEXTALIGN_LEFT,
):
    """
    Renders bounding boxes, label, and confidence on an image if confidence is above the threshold.
    This works with the output of the batch processing API.
    Supports classification, if the detection contains classification results according to the
    API output version 1.0.
    Args:
        detections: detections on the image, example content:
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                }
            ]
            ...where the bbox coordinates are [x, y, box_width, box_height].
            (0, 0) is the upper-left.  Coordinates are normalized.
            Supports classification results, if *detections* has the format
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                    "classifications": [
                        ["3", 0.901],
                        ["1", 0.071],
                        ["4", 0.025]
                    ]
                }
            ]
        image: PIL.Image object
        label_map: optional, mapping the numerical label to a string name. The type of the numerical label
            (default string) needs to be consistent with the keys in label_map; no casting is carried out.
            If this is None, no labels are shown.
        classification_label_map: optional, mapping of the string class labels to the actual class names.
            The type of the numerical label (default string) needs to be consistent with the keys in
            label_map; no casting is carried out.  If this is None, no classification labels are shown.
        confidence_threshold: optional, threshold above which the bounding box is rendered.
        thickness: line thickness in pixels. Default value is 4.
        expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
        classification_confidence_threshold: confidence above which classification result is retained.
        max_classifications: maximum number of classification results retained for one image.
    image is modified in place.
    """

    display_boxes = []
    display_strs = []  # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    classes = []  # for color selection

    for detection in detections:

        score = detection["conf"]
        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection["bbox"]
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
            clss = detection["category"]

            # {} is the default, which means "show labels with no mapping", so don't use "if label_map" here
            # if label_map:
            if label_map is not None:
                label = label_map[clss] if clss in label_map else clss
                displayed_label = ["{}: {}%".format(label, round(100 * score))]
            else:
                displayed_label = ""

            if "classifications" in detection:

                # To avoid duplicate colors with detection-only visualization, offset
                # the classification class index by the number of detection classes
                clss = NUM_DETECTOR_CATEGORIES + int(detection["classifications"][0][0])
                classifications = detection["classifications"]
                if len(classifications) > max_classifications:
                    classifications = classifications[0:max_classifications]

                for classification in classifications:

                    p = classification[1]
                    if p < classification_confidence_threshold:
                        continue
                    class_key = classification[0]
                    if (classification_label_map is not None) and (class_key in classification_label_map):
                        class_name = classification_label_map[class_key]
                    else:
                        class_name = class_key
                    displayed_label += ["{}: {:5.1%}".format(class_name.lower(), classification[1])]

                # ...for each classification

            # ...if we have classification results

            display_strs.append(displayed_label)
            classes.append(clss)

        # ...if the confidence of this detection is above threshold

    # ...for each detection

    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(
        image,
        display_boxes,
        classes,
        display_strs=display_strs,
        thickness=thickness,
        expansion=expansion,
        colormap=colormap,
        textalign=textalign,
    )


def draw_bounding_boxes_on_image(
    image, boxes, classes, thickness=4, expansion=0, display_strs=None, colormap=COLORS, textalign=TEXTALIGN_LEFT
):
    """
    Draws bounding boxes on an image.
    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      classes: a list of ints or strings (that can be cast to ints) corresponding to the class labels of the boxes.
             This is only used for selecting the color to render the bounding box in.
      thickness: line thickness in pixels. Default value is 4.
      expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
      display_strs: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        # print('Input must be of size [N, 4], but is ' + str(boxes_shape))
        return  # no object detection on this image, return
    for i in range(boxes_shape[0]):
        if display_strs:
            display_str_list = display_strs[i]
            draw_bounding_box_on_image(
                image,
                boxes[i, 0],
                boxes[i, 1],
                boxes[i, 2],
                boxes[i, 3],
                classes[i],
                thickness=thickness,
                expansion=expansion,
                display_str_list=display_str_list,
                colormap=colormap,
                textalign=textalign,
            )


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    clss=None,
    thickness=4,
    expansion=0,
    display_str_list=(),
    use_normalized_coordinates=True,
    label_font_size=28,
    colormap=COLORS,
    textalign=TEXTALIGN_LEFT,
):
    """
    Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.
    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box - upper left.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    clss: str, the class of the object in this bounding box - will be cast to an int.
    thickness: line thickness. Default value is 4.
    expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
    display_str_list: list of strings to display in box
        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    label_font_size: font size to attempt to load arial.ttf with
    """
    if clss is None:
        color = colormap[1]
    else:
        color = colormap[int(clss) % len(colormap)]

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    if expansion > 0:

        left -= expansion
        right += expansion
        top -= expansion
        bottom += expansion

        # Deliberately trimming to the width of the image only in the case where
        # box expansion is turned on.  There's not an obvious correct behavior here,
        # but the thinking is that if the caller provided an out-of-range bounding
        # box, they meant to do that, but at least in the eyes of the person writing
        # this comment, if you expand a box for visualization reasons, you don't want
        # to end up with part of a box.
        #
        # A slightly more sophisticated might check whether it was in fact the expansion
        # that made this box larger than the image, but this is the case 99.999% of the time
        # here, so that doesn't seem necessary.
        left = max(left, 0)
        right = max(right, 0)
        top = max(top, 0)
        bottom = max(bottom, 0)

        left = min(left, im_width - 1)
        right = min(right, im_width - 1)
        top = min(top, im_height - 1)
        bottom = min(bottom, im_height - 1)

    # ...if we need to expand boxes

    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype("roboto.ttf", label_font_size)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:

        # Skip empty strings
        if len(display_str) == 0:
            continue

        text_width, text_height = font.getsize(display_str)

        text_left = left

        if textalign == TEXTALIGN_RIGHT:
            text_left = right - text_width

        margin = np.ceil(0.05 * text_height)

        draw.rectangle(
            [(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width, text_bottom)], fill=color
        )

        draw.text((text_left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)

        text_bottom -= text_height + 2 * margin


# ==========================================================================================
