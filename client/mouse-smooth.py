import tkinter as tk

class MouseSmoothingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Smoothing")
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        self.points = [] 

        self.canvas.bind("<Motion>", self.on_mouse_motion)
        self.smoothed_points = []  # List to store smoothed points

        self.update_smoothed_points()  # Start updating smoothed points

    def on_mouse_motion(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))

    def update_smoothed_points(self):
        if len(self.points) > 1:  # Ensure there are at least two points for smoothing
            smoothed_x, smoothed_y = self.smooth_points(self.points[-2:], smoothing_factor=0.2)
            self.smoothed_points.append(smoothed_x)
            self.smoothed_points.append(smoothed_y)

        self.canvas.delete("line")  # Clear the previous line
        if len(self.smoothed_points) >= 4:  # Ensure there are enough points to draw a line
            self.canvas.create_line(self.smoothed_points, tags="line")  # Draw the smoothed line

        self.root.after(10, self.update_smoothed_points)  # Schedule the next update

    @staticmethod
    def smooth_points(points, smoothing_factor=0.2):
        smoothed_x = (1 - smoothing_factor) * points[0][0] + smoothing_factor * points[1][0]
        smoothed_y = (1 - smoothing_factor) * points[0][1] + smoothing_factor * points[1][1]
        return smoothed_x, smoothed_y

if __name__ == "__main__":
    root = tk.Tk()
    app = MouseSmoothingApp(root)
    root.mainloop()
