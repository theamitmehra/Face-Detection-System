import cv2
import os
from pathlib import Path
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, PhotoImage, Toplevel, Frame
from tkinter.ttk import Progressbar, Style
from PIL import Image, ImageTk
import logging
from tqdm import tqdm


COLORS = {
    'bg': '#1e1e1e',
    'fg': '#ffffff',
    'button_bg': '#2d2d2d',
    'button_hover': '#3d3d3d',
    'accent': '#007acc'
}

class CustomButton(Button):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=COLORS['button_bg'],
            fg=COLORS['fg'],
            activebackground=COLORS['button_hover'],
            activeforeground=COLORS['fg'],
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=10,
            font=('Arial', 11),
            cursor='hand2'
        )
        
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self.configure(background=COLORS['button_hover'])

    def on_leave(self, e):
        self.configure(background=COLORS['button_bg'])

class FaceDetector:
    def __init__(self, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Changed to green for better visibility
        return img, len(faces)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_files = list(Path(input_dir).glob('*.jpg')) + \
                     list(Path(input_dir).glob('*.jpeg')) + \
                     list(Path(input_dir).glob('*.png'))

        if not image_files:
            return "No images found in the directory.", []

        processed_images = []
        for img_path in tqdm(image_files, desc="Processing images"):
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Could not read image: {img_path}")
                continue

            processed_img, face_count = self.detect_faces(img)
            output_path = Path(output_dir) / f'processed_{img_path.name}'
            cv2.imwrite(str(output_path), processed_img)
            processed_images.append(output_path)

        return f"Processing completed! Processed {len(processed_images)} images.", processed_images

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.setup_window()
        self.detector = FaceDetector()
        self.input_dir = None
        self.output_dir = None
        self.processed_images = []
        
        self.setup_styles()
        self.create_widgets()

    def setup_window(self):
        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size to full screen
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Configure the main background
        self.root.configure(bg=COLORS['bg'])
        
        # Create main container
        self.main_container = Frame(self.root, bg=COLORS['bg'])
        self.main_container.pack(expand=True, fill='both', padx=50, pady=50)

    def setup_styles(self):
        # Configure style for progressbar
        style = Style()
        style.theme_use('default')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=COLORS['button_bg'],
            background=COLORS['accent'],
            darkcolor=COLORS['accent'],
            lightcolor=COLORS['accent']
        )

    def create_widgets(self):
        # Title
        title_label = Label(
            self.main_container,
            text="Face Detection Application",
            font=("Arial", 24, "bold"),
            bg=COLORS['bg'],
            fg=COLORS['fg']
        )
        title_label.pack(pady=30)

        # Buttons container
        button_frame = Frame(self.main_container, bg=COLORS['bg'])
        button_frame.pack(pady=20)

        # Buttons
        self.select_input_btn = CustomButton(
            button_frame,
            text="Select Input Folder",
            command=self.select_input_folder
        )
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = CustomButton(
            button_frame,
            text="Select Output Folder",
            command=self.select_output_folder
        )
        self.select_output_btn.pack(pady=10)

        self.process_btn = CustomButton(
            button_frame,
            text="Process Images",
            command=self.process_images,
            state="disabled"
        )
        self.process_btn.pack(pady=10)

        self.view_images_btn = CustomButton(
            button_frame,
            text="View Processed Images",
            command=self.view_images,
            state="disabled"
        )
        self.view_images_btn.pack(pady=10)

        # Progress bar
        self.progress = Progressbar(
            self.main_container,
            style="Custom.Horizontal.TProgressbar",
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress.pack(pady=30)

        # Status label
        self.status_label = Label(
            self.main_container,
            text="",
            font=("Arial", 12),
            bg=COLORS['bg'],
            fg=COLORS['fg']
        )
        self.status_label.pack(pady=10)

    def select_input_folder(self):
        self.input_dir = filedialog.askdirectory(title="Select Input Folder")
        if self.input_dir:
            self.status_label.config(text=f"Input folder: {self.input_dir}")
            self.check_ready_to_process()

    def select_output_folder(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Folder")
        if self.output_dir:
            self.status_label.config(text=f"Output folder: {self.output_dir}")
            self.check_ready_to_process()

    def check_ready_to_process(self):
        if self.input_dir and self.output_dir:
            self.process_btn.config(state="normal")

    def process_images(self):
        if not self.input_dir or not self.output_dir:
            messagebox.showwarning("Error", "Please select both input and output folders!")
            return

        self.status_label.config(text="Processing images...")
        self.progress["value"] = 0
        self.root.update_idletasks()

        result, self.processed_images = self.detector.process_directory(self.input_dir, self.output_dir)

        self.progress["value"] = 100
        self.status_label.config(text=result)
        self.view_images_btn.config(state="normal")
        messagebox.showinfo("Processing Complete", result)

    def create_image_viewer(self, viewer):
        viewer.configure(bg=COLORS['bg'])
        # Set viewer to full screen
        screen_width = viewer.winfo_screenwidth()
        screen_height = viewer.winfo_screenheight()
        viewer.geometry(f"{screen_width}x{screen_height}+0+0")

        main_frame = Frame(viewer, bg=COLORS['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        canvas = Canvas(
            main_frame,
            bg=COLORS['bg'],
            highlightthickness=0
        )
        canvas.pack(expand=True, fill='both', pady=20)

        button_frame = Frame(main_frame, bg=COLORS['bg'])
        button_frame.pack(pady=20)

        return canvas, button_frame

    def view_images(self):
        if not self.processed_images:
            messagebox.showwarning("No Images", "No processed images to display!")
            return

        viewer = Toplevel(self.root)
        viewer.title("Processed Images Viewer")
        
        canvas, button_frame = self.create_image_viewer(viewer)
        
        idx = [0]

        def show_image():
            img_path = self.processed_images[idx[0]]
            img = Image.open(img_path)
            
            # Calculate scaling to fit the canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            img_width, img_height = img.size
            
            scale = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            canvas.delete("all")
            canvas.create_image(
                canvas_width//2,
                canvas_height//2,
                anchor="center",
                image=img_tk
            )
            canvas.image = img_tk
            viewer.title(f"Processed Image Viewer - {Path(img_path).name}")

        def next_image():
            if idx[0] < len(self.processed_images) - 1:
                idx[0] += 1
                show_image()

        def prev_image():
            if idx[0] > 0:
                idx[0] -= 1
                show_image()

        prev_btn = CustomButton(button_frame, text="Previous", command=prev_image)
        prev_btn.pack(side="left", padx=10)
        
        next_btn = CustomButton(button_frame, text="Next", command=next_image)
        next_btn.pack(side="left", padx=10)

        # Wait for the window to be drawn before showing the first image
        viewer.update()
        show_image()

        # Bind resize event to update image size
        canvas.bind('<Configure>', lambda e: show_image())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root = Tk()
    app = FaceDetectionApp(root)
    root.mainloop()