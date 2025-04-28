import cv2
import numpy as np

class LaneFollower:
    def __init__(self):
        pass

    def crop_bottom_half(self, image, ratio=0.5):
        height = image.shape[0]
        start_row = int(height * (1 - ratio))
        return image[start_row:, :]

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_image = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    def isolate_white(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        return cv2.inRange(hsv, lower_white, upper_white)
    
    def line_length(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def is_horizontal_or_vertical(self, x1, y1, x2, y2, threshold=0.15):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx == 0 or dx < threshold * dy:
            return True  # Vertical
        if dy == 0 or dy < threshold * dx:
            return True  # Horizontal
        if self.line_length(x1, y1, x2, y2) < 1200:  # Filter out lines shorter than 50px
            return True
        return False

    def process_image(self, image):
        enhanced_image = self.enhance_contrast(image)  # Apply contrast enhancement
        white_mask = self.isolate_white(enhanced_image)

        # Apply morphological operations to clean up the white mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)


        edges = cv2.Canny(cleaned, 30, 100)
        height, width = edges.shape

        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.9), 0),
            (int(width * 0.1), 0),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=300)

        output_image = np.copy(image)
        lane_center = width // 2

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if self.is_horizontal_or_vertical(x1, y1, x2, y2):
                        continue
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if slope < -0.2:
                        left_lines.append((x1, y1, x2, y2))
                    elif slope > 0.2:
                        right_lines.append((x1, y1, x2, y2))

            def average_x(lines):
                if not lines:
                    return None
                return np.mean([x1 + x2 for x1, _, x2, _ in lines]) / 2

            left_x = average_x(left_lines)
            right_x = average_x(right_lines)

            if left_x is not None and right_x is not None:
                lane_center = (left_x + right_x) / 2

            for x1, y1, x2, y2 in left_lines + right_lines:
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if left_x is not None:
                cv2.line(output_image, (int(left_x), 0), (int(left_x), height), (255, 0, 0), 2)
            if right_x is not None:
                cv2.line(output_image, (int(right_x), 0), (int(right_x), height), (255, 0, 0), 2)

        cv2.line(output_image, (int(lane_center), 0), (int(lane_center), height), (0, 0, 255), 2)
        cv2.putText(output_image, f"Lane center: {int(lane_center)}",
                    (int(lane_center) + 10, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(output_image, (int(lane_center), height - 30), 10, (255, 0, 0), -1)

        return output_image, lane_center

    def calculate_steering_angle(self, lane_center, image_width):
        error = lane_center - image_width // 2
        max_error = image_width // 2
        return error / max_error

    def run(self, image_path):
        # Read the test image
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print("Error: Image not found")
            return

        cropped = self.crop_bottom_half(cv_image)
        processed_image, lane_center = self.process_image(cropped)
        angle = self.calculate_steering_angle(lane_center, cropped.shape[1])

        print(f'Steering Angle: {angle}')

        # Show the processed image
        cv2.imshow("Lane Detection", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    lane_follower = LaneFollower()
    lane_follower.run('test_image.jpg')  # Specify your test image here
