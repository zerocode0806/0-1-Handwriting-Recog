import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox
import os

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.load_saved_model()
        
    def load_saved_model(self):
        """Load the model trained by neural.py (.keras only, same folder as script)"""

        # Get the folder where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_keras = os.path.join(base_dir, 'trained_digit_model_cnn.keras') # change (#) to cnn or dense as needed

        if os.path.exists(model_path_keras):
            try:
                self.model = tf.keras.models.load_model(model_path_keras)
                print("✓ Successfully loaded trained model (.keras) from neural.py!")
                return True
            except Exception as e:
                print(f"⚠️ Error loading .keras model: {e}")
                return False
        else:
            print("❌ No .keras model found!")
            print("Please run neural.py first to train and save the model.")
            return False



    def predict_digit(self, image_array):
        """Predict digit from image array"""
        if self.model is None:
            return None, []

        # Reshape and normalize for CNN input (28x28x1)
        image_array = image_array.reshape(1, 28, 28, 1).astype("float32") / 255.0

        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)[0]
        predicted_digit = np.argmax(predictions)

        return predicted_digit, predictions

class DigitRecognitionGUI:
    def __init__(self):
        self.recognizer = DigitRecognizer()
        if self.recognizer.model is not None:
            self.setup_gui()
        else:
            self.show_error_message()
            
    def show_error_message(self):
        """Show error if no model found"""
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "No Model Found", 
            "Please run neural.py first to train and save the model!"
        )
        root.destroy()
        
    def setup_gui(self):
        """Setup the minimalist GUI"""
        self.root = tk.Tk()
        self.root.title("Digit Recognition - Neural Network Output")
        self.root.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Handwritten Digit Recognition", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Drawing section
        draw_frame = ttk.LabelFrame(main_frame, text="Draw a Digit", padding="15")
        draw_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Canvas for drawing
        self.canvas = tk.Canvas(draw_frame, width=250, height=250, bg='black', cursor='pencil')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        # Drawing controls
        ttk.Button(draw_frame, text="Clear", 
                  command=self.clear_canvas).grid(row=1, column=0, padx=(0, 5))
        ttk.Button(draw_frame, text="Predict", 
                  command=self.predict_drawing).grid(row=1, column=1, padx=(5, 0))
        
        # Neural network output section
        neural_frame = ttk.LabelFrame(main_frame, text="Neural Network Output", padding="15")
        neural_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Create neuron display (2 rows of 5)
        self.neuron_frames = []
        self.neuron_labels = []
        self.prediction_labels = []
        
        for i in range(10):
            row = i // 5
            col = i % 5
            
            # Frame for each neuron
            neuron_frame = ttk.Frame(neural_frame, relief='solid', borderwidth=1, padding="8")
            neuron_frame.grid(row=row, column=col, padx=3, pady=3, sticky=(tk.W, tk.E))
            self.neuron_frames.append(neuron_frame)
            
            # Digit number (large)
            digit_label = ttk.Label(neuron_frame, text=str(i), 
                                   font=('Arial', 24, 'bold'), 
                                   anchor='center')
            digit_label.grid(row=0, column=0, pady=(0, 5))
            self.neuron_labels.append(digit_label)
            
            # Prediction percentage
            pred_label = ttk.Label(neuron_frame, text="0.0%", 
                                  font=('Arial', 10), 
                                  anchor='center')
            pred_label.grid(row=1, column=0)
            self.prediction_labels.append(pred_label)
            
            # Configure column weight
            neuron_frame.columnconfigure(0, weight=1)
        
        # Result display
        self.result_label = ttk.Label(main_frame, 
                                     text="Draw a digit above and click Predict", 
                                     font=('Arial', 14), 
                                     anchor='center')
        self.result_label.grid(row=2, column=0, columnspan=2, pady=(20, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Initialize display
        self.update_neural_display(None)
        
    def update_neural_display(self, predictions):
        """Update the neural network output display"""
        if predictions is None:
            # Reset all displays
            for i in range(10):
                self.prediction_labels[i].config(text="0.0%", foreground='gray')
                self.neuron_labels[i].config(foreground='gray')
                self.neuron_frames[i].config(relief='solid')
        else:
            # Find the highest prediction
            max_digit = np.argmax(predictions)
            max_confidence = predictions[max_digit]
            
            # Update each neuron's display
            for i in range(10):
                confidence = predictions[i] * 100
                self.prediction_labels[i].config(text=f"{confidence:.1f}%")
                
                # Highlight the winning neuron
                if i == max_digit:
                    self.neuron_labels[i].config(foreground='green')
                    self.prediction_labels[i].config(foreground='green')
                    self.neuron_frames[i].config(relief='solid', borderwidth=3)
                else:
                    # Dim the other neurons based on their confidence
                    if confidence < 1.0:
                        color = 'lightgray'
                    else:
                        color = 'black'
                    self.neuron_labels[i].config(foreground=color)
                    self.prediction_labels[i].config(foreground=color)
                    self.neuron_frames[i].config(relief='solid', borderwidth=1)
            
            # Update result
            self.result_label.config(text=f"Prediction: {max_digit} (Confidence: {max_confidence*100:.1f}%)")
        
    def paint(self, event):
        """Handle drawing on canvas"""
        x, y = event.x, event.y
        r = 15  # Brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.result_label.config(text="Canvas cleared - draw a new digit")
        self.update_neural_display(None)
    
    def predict_drawing(self):
        """Predict the drawn digit"""
        try:
            # Convert canvas to image
            canvas_items = self.canvas.find_all()
            if not canvas_items:
                self.result_label.config(text="Please draw something first!")
                self.update_neural_display(None)
                return
            
            # Create image from canvas
            img = Image.new('L', (250, 250), 0)  # Black background
            draw = ImageDraw.Draw(img)
            
            # Draw all canvas items
            for item in canvas_items:
                coords = self.canvas.coords(item)
                if len(coords) == 4:  # oval
                    draw.ellipse(coords, fill=255)  # White fill
            
            # Resize to 28x28
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            
            # Predict
            digit, predictions = self.recognizer.predict_digit(img_array)
            if digit is not None:
                # Update neural network display
                self.update_neural_display(predictions)
            else:
                self.result_label.config(text="Could not make prediction")
                self.update_neural_display(None)
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            self.result_label.config(text="Error - try redrawing")
            self.update_neural_display(None)
    
    def run(self):
        """Start the GUI"""
        if hasattr(self, 'root'):
            self.root.mainloop()

if __name__ == "__main__":
    print("Loading Minimalist Digit Recognition App...")
    print("Looking for trained model from neural.py...")
    
    app = DigitRecognitionGUI()
    if hasattr(app, 'root'):
        app.run()