import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RemoveAnythingVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Remove Anything Video GUI")
        
        # Variables
        self.video_path = tk.StringVar()
        self.point_coords = []
        self.current_frame = None
        self.frame_display = None
        
        # Default values for other parameters
        self.dilate_kernel_size = 15
        self.sam_model_type = "vit_h"
        self.sam_ckpt = "./pretrained_models/sam_vit_h_4b8939.pth"
        self.lama_config = "lama/configs/prediction/default.yaml"
        self.lama_ckpt = "./pretrained_models/big-lama"
        self.tracker_ckpt = "vitb_384_mae_ce_32x4_ep300"
        self.vi_ckpt = "./pretrained_models/sttn.pth"
        self.mask_idx = 2
        self.fps = 25
        self.output_dir = "./results"
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video selection
        video_frame = tk.Frame(main_frame)
        video_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(video_frame, text="Input Video:").pack(side=tk.LEFT)
        tk.Entry(video_frame, textvariable=self.video_path, width=50).pack(side=tk.LEFT, padx=5)
        tk.Button(video_frame, text="Browse...", command=self.browse_video).pack(side=tk.LEFT)
        
        # Frame display
        self.frame_display = tk.Label(main_frame, text="Select a video to display the first frame", 
                                     bg="black", fg="white", height=20)
        self.frame_display.pack(fill=tk.BOTH, expand=True, pady=10)
        self.frame_display.bind("<Button-1>", self.on_frame_click)
        
        # Coordinates display
        self.coords_label = tk.Label(main_frame, text="Click on the frame to select a point")
        self.coords_label.pack(pady=5)
        
        # Run button
        tk.Button(main_frame, text="Run Remove Anything", command=self.run_remove_anything).pack(pady=10)
        
        # Advanced settings button
        tk.Button(main_frame, text="Advanced Settings", command=self.show_advanced_settings).pack(pady=5)
        
    def browse_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if video_path:
            self.video_path.set(video_path)
            self.load_first_frame()
    
    def load_first_frame(self):
        try:
            cap = cv2.VideoCapture(self.video_path.get())
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert from BGR to RGB
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize the frame if it's too large
                h, w = self.current_frame.shape[:2]
                max_size = 800
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    self.current_frame = cv2.resize(self.current_frame, (new_w, new_h))
                
                # Convert to PIL Image and then to PhotoImage
                pil_img = Image.fromarray(self.current_frame)
                img_tk = ImageTk.PhotoImage(image=pil_img)
                
                # Update the label
                self.frame_display.config(image=img_tk, width=pil_img.width, height=pil_img.height)
                self.frame_display.image = img_tk  # Keep a reference
                
                # Reset point coordinates
                self.point_coords = []
                self.coords_label.config(text="Click on the frame to select a point")
            else:
                messagebox.showerror("Error", "Could not read the video file.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
    
    def on_frame_click(self, event):
        if self.current_frame is not None:
            # Store the clicked coordinates
            self.point_coords = [event.x, event.y]
            
            # Update the coordinates label
            self.coords_label.config(text=f"Selected point: ({event.x}, {event.y})")
            
            # Create a copy of the frame to draw on
            frame_copy = self.current_frame.copy()
            
            # Draw a circle at the clicked point
            cv2.circle(frame_copy, (event.x, event.y), 5, (255, 0, 0), -1)
            
            # Convert to PIL Image and then to PhotoImage
            pil_img = Image.fromarray(frame_copy)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            
            # Update the label
            self.frame_display.config(image=img_tk)
            self.frame_display.image = img_tk  # Keep a reference
    
    def show_advanced_settings(self):
        # Create a new window for advanced settings
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Advanced Settings")
        settings_window.geometry("500x600")
        
        # Create a frame with scrollbar
        canvas = tk.Canvas(settings_window)
        scrollbar = tk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Output directory
        output_dir_var = tk.StringVar(value=self.output_dir)
        tk.Label(scrollable_frame, text="Output Directory:").pack(anchor="w", padx=10, pady=(10, 0))
        output_dir_frame = tk.Frame(scrollable_frame)
        output_dir_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(output_dir_frame, textvariable=output_dir_var, width=30).pack(side="left", padx=(0, 5))
        tk.Button(output_dir_frame, text="Browse...", 
                 command=lambda: output_dir_var.set(filedialog.askdirectory())).pack(side="left")
        
        # SAM model type
        sam_model_var = tk.StringVar(value=self.sam_model_type)
        tk.Label(scrollable_frame, text="SAM Model Type:").pack(anchor="w", padx=10, pady=(10, 0))
        sam_model_frame = tk.Frame(scrollable_frame)
        sam_model_frame.pack(fill="x", padx=10, pady=(0, 10))
        for model_type in ["vit_h", "vit_l", "vit_b", "vit_t"]:
            tk.Radiobutton(sam_model_frame, text=model_type, variable=sam_model_var, value=model_type).pack(side="left")
        
        # SAM checkpoint
        sam_ckpt_var = tk.StringVar(value=self.sam_ckpt)
        tk.Label(scrollable_frame, text="SAM Checkpoint:").pack(anchor="w", padx=10, pady=(10, 0))
        sam_ckpt_frame = tk.Frame(scrollable_frame)
        sam_ckpt_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(sam_ckpt_frame, textvariable=sam_ckpt_var, width=30).pack(side="left", padx=(0, 5))
        tk.Button(sam_ckpt_frame, text="Browse...", 
                 command=lambda: sam_ckpt_var.set(filedialog.askopenfilename())).pack(side="left")
        
        # LAMA config
        lama_config_var = tk.StringVar(value=self.lama_config)
        tk.Label(scrollable_frame, text="LAMA Config:").pack(anchor="w", padx=10, pady=(10, 0))
        lama_config_frame = tk.Frame(scrollable_frame)
        lama_config_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(lama_config_frame, textvariable=lama_config_var, width=30).pack(side="left", padx=(0, 5))
        tk.Button(lama_config_frame, text="Browse...", 
                 command=lambda: lama_config_var.set(filedialog.askopenfilename())).pack(side="left")
        
        # LAMA checkpoint
        lama_ckpt_var = tk.StringVar(value=self.lama_ckpt)
        tk.Label(scrollable_frame, text="LAMA Checkpoint:").pack(anchor="w", padx=10, pady=(10, 0))
        lama_ckpt_frame = tk.Frame(scrollable_frame)
        lama_ckpt_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(lama_ckpt_frame, textvariable=lama_ckpt_var, width=30).pack(side="left", padx=(0, 5))
        tk.Button(lama_ckpt_frame, text="Browse...", 
                 command=lambda: lama_ckpt_var.set(filedialog.askdirectory())).pack(side="left")
        
        # Tracker checkpoint
        tracker_ckpt_var = tk.StringVar(value=self.tracker_ckpt)
        tk.Label(scrollable_frame, text="Tracker Checkpoint:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Entry(scrollable_frame, textvariable=tracker_ckpt_var).pack(fill="x", padx=10, pady=(0, 10))
        
        # Video inpainter checkpoint
        vi_ckpt_var = tk.StringVar(value=self.vi_ckpt)
        tk.Label(scrollable_frame, text="Video Inpainter Checkpoint:").pack(anchor="w", padx=10, pady=(10, 0))
        vi_ckpt_frame = tk.Frame(scrollable_frame)
        vi_ckpt_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Entry(vi_ckpt_frame, textvariable=vi_ckpt_var, width=30).pack(side="left", padx=(0, 5))
        tk.Button(vi_ckpt_frame, text="Browse...", 
                 command=lambda: vi_ckpt_var.set(filedialog.askopenfilename())).pack(side="left")
        
        # Mask index
        mask_idx_var = tk.IntVar(value=self.mask_idx)
        tk.Label(scrollable_frame, text="Mask Index:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Spinbox(scrollable_frame, from_=0, to=10, textvariable=mask_idx_var).pack(fill="x", padx=10, pady=(0, 10))
        
        # FPS
        fps_var = tk.IntVar(value=self.fps)
        tk.Label(scrollable_frame, text="FPS:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Spinbox(scrollable_frame, from_=1, to=60, textvariable=fps_var).pack(fill="x", padx=10, pady=(0, 10))
        
        # Dilate kernel size
        dilate_kernel_var = tk.IntVar(value=self.dilate_kernel_size)
        tk.Label(scrollable_frame, text="Dilate Kernel Size:").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Spinbox(scrollable_frame, from_=0, to=50, textvariable=dilate_kernel_var).pack(fill="x", padx=10, pady=(0, 10))
        
        # Save button
        def save_settings():
            self.output_dir = output_dir_var.get()
            self.sam_model_type = sam_model_var.get()
            self.sam_ckpt = sam_ckpt_var.get()
            self.lama_config = lama_config_var.get()
            self.lama_ckpt = lama_ckpt_var.get()
            self.tracker_ckpt = tracker_ckpt_var.get()
            self.vi_ckpt = vi_ckpt_var.get()
            self.mask_idx = mask_idx_var.get()
            self.fps = fps_var.get()
            self.dilate_kernel_size = dilate_kernel_var.get()
            settings_window.destroy()
            messagebox.showinfo("Settings", "Settings have been saved!")
        
        tk.Button(scrollable_frame, text="Save Settings", command=save_settings).pack(pady=20)
    
    def run_remove_anything(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
        
        if not self.point_coords:
            messagebox.showerror("Error", "Please click on the frame to select a point")
            return
        
        # Prepare command
        cmd = [
            "python", "remove_anything_video.py",
            "--input_video", self.video_path.get(),
            "--coords_type", "key_in",
            "--point_coords", str(self.point_coords[0]), str(self.point_coords[1]),
            "--point_labels", "1",
            "--dilate_kernel_size", str(self.dilate_kernel_size),
            "--output_dir", self.output_dir,
            "--sam_model_type", self.sam_model_type,
            "--sam_ckpt", self.sam_ckpt,
            "--lama_config", self.lama_config,
            "--lama_ckpt", self.lama_ckpt,
            "--tracker_ckpt", self.tracker_ckpt,
            "--vi_ckpt", self.vi_ckpt,
            "--mask_idx", str(self.mask_idx),
            "--fps", str(self.fps)
        ]
        
        # Run in a separate thread to avoid blocking the UI
        import threading
        
        def run_process():
            try:
                # Change to the project root directory
                original_dir = os.getcwd()
                os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                # Change back to original directory
                os.chdir(original_dir)
                
                if process.returncode == 0:
                    messagebox.showinfo("Success", "Video processing completed successfully!")
                else:
                    messagebox.showerror("Error", f"Error during processing:\n{stderr.decode()}")
            except Exception as e:
                messagebox.showerror("Error", f"Error running command: {str(e)}")
        
        threading.Thread(target=run_process).start()
        messagebox.showinfo("Processing", "Video processing has started in the background. This might take some time.")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = RemoveAnythingVideoGUI(root)
    root.mainloop() 