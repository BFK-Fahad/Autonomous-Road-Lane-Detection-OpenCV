import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Global Variables
deviation = 0
frame_idx = 0
lanes_detected = False

# Video Input
cap = cv2.VideoCapture('C:\\Users\\HP\\Downloads\\Road_video.mp4')
#cap = cv2.VideoCapture(3)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Image Processing Preparations
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #frame = cv2.flip(frame, -1)
    frame = cv2.resize(frame, (1100, 550))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 50, 150)

    # Region of Interest Masking
    height = canny.shape[0]
    mask = np.zeros_like(canny)
    triangle = np.array([[(-550, height), (1400, height), (450, 280)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)

    # Line Detection
    lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    left_fit = []
    right_fit = []
    left_fit_average = None
    right_fit_average = None
    lanes_detected = False

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.ravel()

            if abs(x2 - x1) < 1e-6:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)

    line_image = np.zeros_like(frame)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_center = int(frame_width / 2)
    int_deviation_data = 5000  # The by default 5000 refers if we want to stop the robot instead of zero or other between in deviation range values can not make the decision for stop
    average_line_center = frame_center  # Defaults to center if no lanes seen

    # Calculate Trajectory and Overlay Lane Graphics
    if left_fit_average is not None and right_fit_average is not None:
        lanes_detected = True
        slope_left, intercept_left = left_fit_average
        slope_right, intercept_right = right_fit_average

        y1 = frame_height
        y2 = int(y1 * 3.0 / 5)

        x1_left = np.clip(int(round((y1 - intercept_left) / slope_left)), 0, frame_width - 1)
        x2_left = np.clip(int(round((y2 - intercept_left) / slope_left)), 0, frame_width - 1)

        x1_right = np.clip(int(round((y1 - intercept_right) / slope_right)), 0, frame_width - 1)
        x2_right = np.clip(int(round((y2 - intercept_right) / slope_right)), 0, frame_width - 1)

        cv2.line(line_image, (x1_left, y1), (x2_left, y2), (255, 0, 0), 10)
        cv2.line(line_image, (x1_right, y1), (x2_right, y2), (255, 0, 0), 10)

        left_line_center = (x1_left + x2_left) / 2
        right_line_center = (x1_right + x2_right) / 2
        average_line_center = int((left_line_center + right_line_center) / 2)

        deviation = average_line_center - frame_center
        int_deviation_data = int(np.clip(deviation, -195, 195))
    elif lanes_detected:
        int_deviation_data = int(deviation)


    # Base display layout blend
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # --- ANIMATED HUD / TELEMETRY OVERLAY ---
    # 1. Target overlay height (30% from bottom of the frame)
    hud_y = int(frame_height * 0.75)

    # 2. Draw Fixed True Center Guide Line (Green)
    cv2.line(combo_image, (frame_center, hud_y - 40), (frame_center, hud_y + 40), (0, 255, 0), 2)
    cv2.putText(combo_image, "CENTER", (frame_center - 30, hud_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if lanes_detected:
        # 3. Draw Dynamic Lane Center Point (Yellow)
        cv2.circle(combo_image, (average_line_center, hud_y), 8, (0, 255, 255), -1)

        # 4. Draw Animated Deviation Connector Vector (Changes color based on severity)
        vector_color = (0, 165, 255) if abs(deviation) > 50 else (0, 255, 255)  # Orange if harsh deviation
        cv2.arrowedLine(combo_image, (frame_center, hud_y), (average_line_center, hud_y), vector_color, 3,
                        tipLength=0.2)

        # 5. Direct steering command text indicators
        if deviation > 5:
            steer_text = f"Steer Right: +{abs(deviation):.1f}"
            text_color = (255, 0, 255)
        elif deviation < -5:
            steer_text = f"Steer Left: -{abs(deviation):.1f}"
            text_color = (255, 0, 255)
        else:
            steer_text = "Tracking Center"
            text_color = (0, 255, 0)

        cv2.putText(combo_image, steer_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    else:
        cv2.putText(combo_image, "Lanes Lost / Searching...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print("----- Deviation ------", int_deviation_data)

    # Final Render
    cv2.imshow('Lane Detection', combo_image)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()