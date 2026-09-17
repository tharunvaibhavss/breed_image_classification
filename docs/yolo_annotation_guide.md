# YOLO Bounding Box Annotation Workflow & Guidelines

This guide details the procedure for generating, verifying, and formatting bounding box annotations for **YOLO Animal Detection** (`class 0 = cattle`, `class 1 = buffalo`).

---

## 📌 Detection Scope & Object Classes

The detection stage isolates animals in images before passing cropped regions to EfficientNet-B0 for breed classification.

| Class ID | Class Name | Description |
|---|---|---|
| `0` | `cattle` | Indigenous cattle breeds (Gir, Ongole, Sahiwal, etc.) |
| `1` | `buffalo` | Indigenous buffalo breeds (Jaffarabadi, Murrah, Surti, etc.) |

---

## 📐 YOLO Label Format Specification

Each image in `data/annotations/yolo/images/{split}/` must have a corresponding label text file with the same base name in `data/annotations/yolo/labels/{split}/{filename}.txt`.

Format of each line:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Constraints:
- `class_id`: Integer (`0` or `1`).
- `x_center`: Normalized X coordinate of box center (`0.0 <= x_center <= 1.0`).
- `y_center`: Normalized Y coordinate of box center (`0.0 <= y_center <= 1.0`).
- `width`: Normalized width of bounding box (`0.0 <= width <= 1.0`).
- `height`: Normalized height of bounding box (`0.0 <= height <= 1.0`).

### Conversion Formula from Pixel Coordinates:
Given image width $W$ and height $H$, and pixel bounding box $(x_{min}, y_{min}, x_{max}, y_{max})$:
$$\text{x\_center} = \frac{x_{min} + x_{max}}{2 \cdot W}$$
$$\text{y\_center} = \frac{y_{min} + y_{max}}{2 \cdot H}$$
$$\text{width} = \frac{x_{max} - x_{min}}{W}$$
$$\text{height} = \frac{y_{max} - y_{min}}{H}$$

---

## 🛠 Recommended Labeling Tools

1. **LabelImg / Labelme**: Free open-source graphical image annotation tool supporting direct export in YOLO format.
2. **CVAT (Computer Vision Annotation Tool)**: Browser-based annotation tool suitable for batch annotation.
3. **Roboflow / Label Studio**: Cloud/local web interface.

---

## ⚠️ Important Rules

- Do NOT invent automated bounding boxes without human review or verified ground truth.
- Do NOT proceed to YOLO model training until annotations exist and pass validation checks (`python -m pytest tests/test_yolo_annotation.py`).
