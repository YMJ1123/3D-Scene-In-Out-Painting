import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import subprocess
import numpy as np
import threading
import queue
import re
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ScrollableErrorDialog:
    def __init__(self, parent, title, message):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("600x400")
        self.dialog.minsize(400, 300)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create main frame with padding
        main_frame = tk.Frame(self.dialog, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create error icon and label
        header_frame = tk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        try:
            # Try to use the standard error icon
            error_icon = tk.Label(header_frame, bitmap="error")
            error_icon.pack(side=tk.LEFT, padx=(0, 10))
        except:
            # Fallback if error icon is not available
            pass
            
        tk.Label(header_frame, text="Error", font=("", 12, "bold")).pack(side=tk.LEFT)
        
        # Create scrolled text widget for the error message
        self.text_widget = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=80, height=20,
            font=("Courier" if os.name == "nt" else "Monospace", 10)
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        self.text_widget.insert(tk.END, message)
        
        # Make text selectable but read-only
        self.text_widget.config(state=tk.DISABLED)
        
        # Enable selection even in disabled state
        self.text_widget.bind("<1>", lambda event: self.text_widget.focus_set())
        
        # Add keyboard shortcuts for copy (Ctrl+C)
        self.text_widget.bind("<Control-c>", self.copy_selection)
        
        # Add a context menu for copy
        self.context_menu = tk.Menu(self.text_widget, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_to_clipboard)
        self.text_widget.bind("<Button-3>", self.show_context_menu)  # Right-click on Windows/Linux
        
        # Add OK button
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add a "Copy to Clipboard" button
        copy_button = tk.Button(button_frame, text="Copy All", command=self.copy_all_text)
        copy_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ok_button = tk.Button(button_frame, text="OK", command=self.dialog.destroy, width=10)
        ok_button.pack(side=tk.RIGHT)
        
        # Set focus on the OK button
        ok_button.focus_set()
        
        # Make dialog modal
        self.dialog.wait_window()
    
    def copy_selection(self, event=None):
        try:
            selected_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(selected_text)
        except tk.TclError:
            # No selection
            pass
        return "break"  # Prevents the event from propagating
    
    def copy_all_text(self):
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(self.text_widget.get(1.0, tk.END))
        
    def copy_to_clipboard(self):
        self.copy_selection()
    
    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

class RemoveAnythingVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Remove Anything Video GUI")
        
        # Variables
        self.video_path = tk.StringVar()
        self.point_coords = []
        self.current_frame = None
        self.frame_display = None
        self.process_queue = queue.Queue()
        self.process_running = False
        
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
    
    def show_error(self, title, message):
        """Show a scrollable error dialog instead of a simple message box"""
        ScrollableErrorDialog(self.root, title, message)
    
    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a paned window to split the UI
        paned_window = tk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for video selection and display
        top_frame = tk.Frame(paned_window)
        paned_window.add(top_frame, stretch="always")
        
        # Video selection
        video_frame = tk.Frame(top_frame)
        video_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(video_frame, text="Input Video:").pack(side=tk.LEFT)
        tk.Entry(video_frame, textvariable=self.video_path, width=50).pack(side=tk.LEFT, padx=5)
        tk.Button(video_frame, text="Browse...", command=self.browse_video).pack(side=tk.LEFT)
        
        # Frame display
        self.frame_display = tk.Label(top_frame, text="Select a video to display the first frame", 
                                     bg="black", fg="white", height=20)
        self.frame_display.pack(fill=tk.BOTH, expand=True, pady=10)
        self.frame_display.bind("<Button-1>", self.on_frame_click)
        
        # Coordinates display
        self.coords_label = tk.Label(top_frame, text="Click on the frame to select a point")
        self.coords_label.pack(pady=5)
        
        # Bottom frame for log display
        bottom_frame = tk.Frame(paned_window, height=150)
        bottom_frame.pack_propagate(False)  # Prevent the frame from shrinking
        paned_window.add(bottom_frame, stretch="always")
        
        # Log display
        log_frame = tk.LabelFrame(bottom_frame, text="Processing Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Run button
        tk.Button(button_frame, text="Run Remove Anything", command=self.run_remove_anything).pack(side=tk.LEFT, padx=5)
        
        # Advanced settings button
        tk.Button(button_frame, text="Advanced Settings", command=self.show_advanced_settings).pack(side=tk.LEFT, padx=5)
        
        # Clear log button
        tk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.RIGHT, padx=5)
    
    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def add_log(self, message):
        """Add a message to the log text widget"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Auto-scroll to the end
        self.log_text.config(state=tk.DISABLED)
    
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
                self.show_error("Error", "Could not read the video file.")
        except Exception as e:
            self.show_error("Error", f"Error loading video: {str(e)}")
    
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
            self.show_error("Error", "Please select a video file")
            return
        
        if not self.point_coords:
            self.show_error("Error", "Please click on the frame to select a point")
            return
        
        # Prepare command
        cmd = [
            # Set environment variables for CUDA and library paths
            # "export", f"LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH", "&&",
            # "export", f"CUDA_HOME=$CONDA_PREFIX", "&&",
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
        
        # Clear the log
        self.clear_log()
        self.add_log("Starting video processing...")
        self.add_log(f"Command: {' '.join(cmd)}")
        
        # Run in a separate thread to avoid blocking the UI
        self.process_running = True
        threading.Thread(target=self.run_process, args=(cmd,)).start()
        
        # Start the output reader thread
        threading.Thread(target=self.process_output_reader).start()
    
    def run_process(self, cmd):
        try:
            # Change to the project root directory
            original_dir = os.getcwd()
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read stdout and stderr in real-time
            for stdout_line in iter(process.stdout.readline, ""):
                self.process_queue.put(("stdout", stdout_line.strip()))
            
            for stderr_line in iter(process.stderr.readline, ""):
                self.process_queue.put(("stderr", stderr_line.strip()))
            
            # Wait for the process to complete
            process.stdout.close()
            process.stderr.close()
            return_code = process.wait()
            
            # Change back to original directory
            os.chdir(original_dir)
            
            # Signal process completion
            if return_code == 0:
                self.process_queue.put(("status", "success"))
            else:
                self.process_queue.put(("status", "error"))
                
        except Exception as e:
            self.process_queue.put(("status", f"exception: {str(e)}"))
        finally:
            self.process_running = False
    
    def process_output_reader(self):
        """Read and display process output from the queue"""
        while self.process_running or not self.process_queue.empty():
            try:
                msg_type, message = self.process_queue.get(timeout=0.1)
                
                if msg_type == "status":
                    if message == "success":
                        self.root.after(0, lambda: self.add_log("\n✅ Processing completed successfully!"))
                        self.root.after(0, lambda: messagebox.showinfo("Success", "Video processing completed successfully!"))
                    elif message.startswith("exception"):
                        error_msg = message.split(":", 1)[1].strip()
                        self.root.after(0, lambda: self.add_log(f"\n❌ Error: {error_msg}"))
                        self.root.after(0, lambda: self.show_error("Error", f"Error running command: {error_msg}"))
                    else:
                        self.root.after(0, lambda: self.add_log("\n❌ Processing failed!"))
                        self.root.after(0, lambda: messagebox.showerror("Error", "Video processing failed!"))
                elif msg_type == "stderr":
                    # Format error messages in red (not actually colored, just prefixed)
                    self.root.after(0, lambda msg=message: self.add_log(f"ERROR: {msg}"))
                else:
                    # Regular stdout messages
                    self.root.after(0, lambda msg=message: self.add_log(msg))
                
                self.process_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in output reader: {str(e)}")
                break

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = RemoveAnythingVideoGUI(root)
    root.mainloop() 