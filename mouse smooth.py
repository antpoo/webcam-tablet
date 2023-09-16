import tkinter as tk

class MouseSmoothingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Smoothing")
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        self.points = [] 

        self.canvas.bind("<Motion>", self.on_mouse_motion)

    def on_mouse_motion(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        if len(self.points) > 1:
            self.draw_smoothed_line()

    def draw_smoothed_line(self):
        smoothed_points = self.smooth_points(self.points)
        self.canvas.delete("line")  # Clear the previous line
        self.canvas.create_line(smoothed_points, tags="line")


    @staticmethod
    def smooth_points(points, smoothing_factor=0.2):
        smoothed_points = []
        for i in range(len(points)):
            if i == 0:
                smoothed_x = points[i][0]
                smoothed_y = points[i][1]
            else:
                smoothed_x = (1 - smoothing_factor) * smoothed_points[-1][0] + smoothing_factor * points[i][0]
                smoothed_y = (1 - smoothing_factor) * smoothed_points[-1][1] + smoothing_factor * points[i][1]
            smoothed_points.append((smoothed_x, smoothed_y))
        return smoothed_points

if __name__ == "__main__":
    root = tk.Tk()
    app = MouseSmoothingApp(root)
    root.mainloop()
