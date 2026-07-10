# Autonomous Road Lane Detection Using OpenCV and Hough Transform

A high-performance Computer Vision pipeline built with **Python, OpenCV, and NumPy** inside the PyCharm IDE. This project tracks autonomous vehicle driving lanes in real-time video streams, calculates the vehicle's spatial deviation from the lane center, and generates dynamic, animated telemetry overlays (steering vector guides).

---

## 📌 Project Features & Image Processing Workflow

The system takes raw driving footage and filters out visual noise using a multi-stage image processing pipeline:

1. **Preprocessing:** Downsamples the video stream frame-by-frame into a standard frame layout $(1100 \times 550)$, converts it to **Grayscale**, and applies a **Gaussian Blur ($7 \times 7$ kernel)** to smooth out high-frequency texture interference like pavement grain.
2. **Canny Edge Detection:** Computes gradient magnitudes to find structural pixel intensity changes, highlighting raw boundaries and edge maps.
3. **Region of Interest (ROI) Masking:** Utilizes a matrix polygon mask (`cv2.fillPoly`) to completely isolate the lower triangular lane driving space, ignoring trees, horizons, and sky.
4. **Probabilistic Hough Line Transform:** Leverages `cv2.HoughLinesP` to extract structural line segments out of isolated edge patterns.
5. **Slope Sorting & Linear Regression:** Automatically groups extracted line arrays into Left Lanes (negative slope) and Right Lanes (positive slope) using `np.polyfit` to generate unified mathematical line equations.
6. **Telemetry & Deviation Calculation:** Computes the mathematical midpoint between the left and right lane coordinates, compares it against the fixed true camera center, and calculates steering vector deviations within a clipped threshold range `[-195, 195]`.

---

## 🧠 Model Pipeline Mechanics

[Raw Input Video Frame]
        │
        ▼
[Resize & Grayscale Conversion]
        │
        ▼
[Gaussian Blur (7x7) -> Canny Edges]
        │
        ▼
[Polygonal ROI Mask (Isolate Lane Space)]
        │
        ▼
[HoughLinesP -> Slope Regression (np.polyfit)]
        │
        ▼
[Calculate Center Midpoint vs Camera Center]
        │
        ▼
[Render Animated HUD & Steering Vector Commands]

---

## 💻 Tech Stack & Tools Used
- **Development Environment:** JetBrains PyCharm IDE
- **Core Library:** OpenCV (`cv2`) for foundational pixel transformations, drawing routines, and matrix functions.
- **Matrix Computation:** NumPy for high-performance algebraic arrays, polynomial fitting (`polyfit`), and numeric clippings.
- **Language:** Python

---

Here is the Project Demo:


https://github.com/user-attachments/assets/6e622606-a3bd-4dcc-885a-3cc7454359fa






## 🛠️ Code Configuration & Core Logic

The backbone tracking array parameters used in the Hough Transform function are tuned to detect stable road lines while dismissing temporary noise fragments:

```python
# Lines parameters configuration snippet
lines = cv2.HoughLinesP(
    masked_image, 
    rho=2, 
    theta=np.pi/180, 
    threshold=100, 
    minLineLength=40, 
    maxLineGap=5
)






