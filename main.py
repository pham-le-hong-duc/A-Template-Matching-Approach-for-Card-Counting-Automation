import os
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from typing import Dict, Set, List, Tuple
import mss
from concurrent.futures import ThreadPoolExecutor

class RealtimeCardDetector:
    def __init__(self, template_base_path: str, threshold: float = 0.85):
        self.template_base_path = template_base_path
        self.threshold = threshold
        
        # Card ordering to match UI display order
        self.SUITS_ORDERED = ['Spade', 'Club', 'Diamond', 'Heart']
        self.RANKS_ORDERED = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
        self.SUIT_SYMBOLS = {'Spade': 'S', 'Heart': 'H', 'Diamond': 'D', 'Club': 'C'}
        
        # Create class mapping
        self.class_names = []
        self.class_to_id = {}
        self._create_class_map()
        
        # Load templates
        self.templates_by_rank = {rank: [] for rank in self.RANKS_ORDERED}
        self.end_template = None  # Special end template
        self._load_templates()
        
        # Thread executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=len(self.RANKS_ORDERED))
        
        # Detected cards set - current frame
        self.detected_cards = set()
        # Memory of all cards ever detected - persistent until reset
        self.detected_cards_memory = set()
        self.detection_lock = threading.Lock()

    def _create_class_map(self):
        """Create class mapping same as original code"""
        class_id_counter = 0
        for suit in self.SUITS_ORDERED:
            for rank in self.RANKS_ORDERED:
                class_name = f"{rank}{self.SUIT_SYMBOLS[suit]}"
                self.class_names.append(class_name)
                label_name = f"{suit}-{rank}"
                self.class_to_id[label_name] = class_id_counter
                class_id_counter += 1

    def _load_templates(self):
        """Load card templates and special end template"""
        print(f"Loading templates from: {self.template_base_path}")
        total_loaded = 0
        
        # Load regular card templates
        for suit in self.SUITS_ORDERED:
            for rank in self.RANKS_ORDERED:
                label = f"{suit}-{rank}"
                file_name = f"{label}.png"
                path = os.path.join(self.template_base_path, file_name)
                
                if not os.path.exists(path):
                    print(f"Warning: Template not found: {path}")
                    continue

                template = cv2.imread(path)
                if template is None:
                    print(f"Warning: Cannot read template: {path}")
                    continue
                    
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                w, h = template_gray.shape[::-1]
                
                self.templates_by_rank[rank].append((template_gray, w, h, label))
                total_loaded += 1
        
        # Load special "End" template
        end_path = os.path.join(self.template_base_path, "End.png")
        if os.path.exists(end_path):
            end_template = cv2.imread(end_path)
            if end_template is not None:
                end_template_gray = cv2.cvtColor(end_template, cv2.COLOR_BGR2GRAY)
                w, h = end_template_gray.shape[::-1]
                self.end_template = (end_template_gray, w, h, "End")
                print("Special 'End' template loaded successfully!")
            else:
                print("Warning: Cannot read End template")
        else:
            print("No 'End.png' template found - reset feature disabled")
                
        print(f"Successfully loaded {total_loaded} card templates.")

    def _match_rank(self, img_gray: np.ndarray, rank_templates: List[Tuple]) -> List[Tuple]:
        """Match templates for a specific rank"""
        candidates = []
        for (template_gray, w, h, label) in rank_templates:
            if img_gray.shape[0] < h or img_gray.shape[1] < w:
                continue

            try:
                res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val >= self.threshold:
                    candidates.append((max_val, max_loc, label, (w, h)))
            except cv2.error:
                pass
        return candidates

    def _find_candidates_parallel(self, img_gray: np.ndarray) -> List[Tuple]:
        """Find card candidates using parallel processing"""
        futures = []
        for rank in self.RANKS_ORDERED:
            rank_templates = self.templates_by_rank[rank]
            if rank_templates:
                futures.append(self.executor.submit(self._match_rank, img_gray, rank_templates))
        
        all_candidates = []
        for future in futures:
            all_candidates.extend(future.result())
            
        return all_candidates

    def _resolve_conflicts(self, candidates: List[Tuple]) -> Set[str]:
        """Resolve conflicts and return detected card labels"""
        detections = {}
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for (max_val, top_left, label, dims) in candidates:
            is_new = True
            for existing_top_left, (ex_val, ex_label, ex_dims) in detections.items():
                if abs(top_left[0] - existing_top_left[0]) < 20 and \
                   abs(top_left[1] - existing_top_left[1]) < 20:
                    is_new = False
                    break
                    
            if is_new:
                detections[top_left] = (max_val, label, dims)
        
        return set(detection[1] for detection in detections.values())

    def _detect_end_template(self, img_gray: np.ndarray) -> bool:
        """Detect special 'End' template in image"""
        if self.end_template is None:
            return False
            
        template_gray, w, h, label = self.end_template
        
        if img_gray.shape[0] < h or img_gray.shape[1] < w:
            return False
            
        try:
            res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Use higher threshold for end template to avoid false positives
            return max_val >= (self.threshold + 0.05)
        except cv2.error:
            return False

    def detect_cards_in_image(self, image: np.ndarray) -> Tuple[Set[str], bool]:
        """Detect cards in the given image, returns (detected_cards, end_detected)"""
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
        
        # Check for end template first
        end_detected = self._detect_end_template(img_gray)
        
        # Detect regular cards
        candidates = self._find_candidates_parallel(img_gray)
        detected_labels = self._resolve_conflicts(candidates)
        
        return detected_labels, end_detected

    def update_detected_cards(self, detected_labels: Set[str], end_detected: bool = False):
        """Update the detected cards set thread-safely with memory"""
        with self.detection_lock:
            # Handle end template detection
            if end_detected:
                print("'End' template detected! Auto-resetting memory...")
                self.detected_cards_memory.clear()
                self.detected_cards.clear()
                return
            
            self.detected_cards = detected_labels.copy()
            # Add newly detected cards to memory
            self.detected_cards_memory.update(detected_labels)
            
            # Auto-reset if all 52 cards detected
            if len(self.detected_cards_memory) >= 52:
                print("All 52 cards detected! Auto-resetting memory...")
                self.detected_cards_memory.clear()

    def get_detected_cards(self) -> Set[str]:
        """Get current detected cards thread-safely"""
        with self.detection_lock:
            return self.detected_cards.copy()
    
    def get_detected_cards_memory(self) -> Set[str]:
        """Get memory of all detected cards thread-safely"""
        with self.detection_lock:
            return self.detected_cards_memory.copy()
    
    def reset_memory(self):
        """Reset the memory of detected cards"""
        with self.detection_lock:
            self.detected_cards_memory.clear()
            self.detected_cards.clear()

class CardUI:
    def __init__(self, detector: RealtimeCardDetector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("Real-time Card Detection")
        self.root.geometry("650x350")
        self.root.configure(bg='#0F4C3A')  # Deep green background
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        # Add window icon effect
        self.root.resizable(True, True)
        
        # Create title label
        title_label = tk.Label(
            self.root,
            text="REAL-TIME CARD DETECTOR",
            font=('Arial', 12, 'bold'),
            bg='#0F4C3A',
            fg='#FFD700',  # Gold color
            pady=5
        )
        title_label.pack(fill=tk.X)
        
        # Create main frame with better styling
        self.main_frame = tk.Frame(
            self.root, 
            bg='#1E6B4F',  # Lighter green
            relief='ridge',
            borderwidth=3
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create card grid
        self.card_labels = {}
        self._create_card_grid()
        
        # Screen capture setup will be done in detection thread
        self.sct = None
        self.monitor = None
        
        # Control variables
        self.running = False
        self.detection_thread = None
        
        # Control buttons
        self._create_controls()
        
        # Start detection automatically
        self.start_detection()

    def _create_card_grid(self):
        """Create the grid of card icons with colors"""
        # Custom order as requested: Heart, Diamond, Club, Spade  
        suits = ['♥', '♦', '♣', '♠']
        # Custom rank order as requested
        ranks = ['3', '4', '5', '6', '7', '8', '9', '10','J', 'Q', 'K', 'A', '2']
        
        # Define colors for suits
        suit_colors = {
            '♠': '#000000',  # Black for Spades
            '♥': '#FF0000',  # Red for Hearts  
            '♦': '#FF6600',  # Orange-Red for Diamonds
            '♣': '#006600'   # Dark Green for Clubs
        }
        
        # Create grid: 4 rows (suits) x 13 columns (ranks)
        for suit_idx, suit in enumerate(suits):
            for rank_idx, rank in enumerate(ranks):
                # Create label for each card
                card_text = f"{rank}{suit}"
                
                # Determine face card styling
                is_face_card = rank in ['J', 'Q', 'K', 'A']
                font_size = 16 if is_face_card else 14
                
                label = tk.Label(
                    self.main_frame,
                    text=card_text,
                    font=('Arial', font_size, 'bold'),
                    bg='#F8F8FF',  # Light gray-white background
                    fg=suit_colors[suit],
                    relief='raised',
                    borderwidth=2,
                    width=4,
                    height=2,
                    cursor='hand2'
                )
                
                # Position in grid
                row = suit_idx
                col = rank_idx
                label.grid(row=row, column=col, padx=1, pady=1, sticky='nsew')
                
                # Store reference with suit-rank format
                suit_name = ['Heart', 'Diamond', 'Club', 'Spade'][suit_idx]
                card_key = f"{suit_name}-{rank}"
                self.card_labels[card_key] = (label, suit_colors[suit])
        
        # Configure grid weights for responsive layout
        for i in range(4):
            self.main_frame.grid_rowconfigure(i, weight=1)
        for i in range(13):
            self.main_frame.grid_columnconfigure(i, weight=1)

    def _create_controls(self):
        """Create control buttons with improved styling"""
        control_frame = tk.Frame(
            self.root, 
            bg='#0F4C3A',
            relief='groove',
            borderwidth=2
        )
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Button styling
        button_config = {
            'font': ('Arial', 10, 'bold'),
            'relief': 'raised',
            'borderwidth': 2,
            'cursor': 'hand2',
            'width': 12
        }
        
        self.start_btn = tk.Button(
            control_frame,
            text="START",
            command=self.start_detection,
            bg='#28A745',  # Bootstrap green
            fg='white',
            activebackground='#218838',
            **button_config
        )
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="STOP",
            command=self.stop_detection,
            bg='#DC3545',  # Bootstrap red
            fg='white',
            activebackground='#C82333',
            **button_config
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.reset_btn = tk.Button(
            control_frame,
            text="RESET",
            command=self.reset_cards,
            bg='#007BFF',  # Bootstrap blue
            fg='white',
            activebackground='#0056B3',
            **button_config
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.clear_memory_btn = tk.Button(
            control_frame,
            text="CLEAR MEM",
            command=self.clear_memory,
            bg='#FFC107',  # Bootstrap yellow/warning
            fg='black',
            activebackground='#E0A800',
            **button_config
        )
        self.clear_memory_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status label with better styling
        self.status_label = tk.Label(
            control_frame,
            text="Status: Stopped",
            bg='#0F4C3A',
            fg='#FFD700',  # Gold color
            font=('Arial', 10, 'bold'),
            relief='sunken',
            borderwidth=1,
            padx=10
        )
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)

    def start_detection(self):
        """Start the detection thread"""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            self.status_label.config(text="Status: Running")
            self.start_btn.config(state='disabled', bg='#6C757D')  # Gray when disabled
            self.stop_btn.config(state='normal', bg='#DC3545')

    def stop_detection(self):
        """Stop the detection thread"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1)
        self.status_label.config(text="Status: Stopped")
        self.start_btn.config(state='normal', bg='#28A745')  # Green when enabled
        self.stop_btn.config(state='disabled', bg='#6C757D')

    def reset_cards(self):
        """Reset all cards memory to bright state"""
        self.detector.reset_memory()
        self._update_card_display()
        print("Card memory reset - all cards back to bright state")

    def clear_memory(self):
        """Clear only the memory, keep current detection running"""
        self.detector.reset_memory()
        self._update_card_display()
        print("Memory cleared - keeping current detection active")

    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        local_sct = None
        try:
            # Create mss instance for this thread
            local_sct = mss.mss()
            
            # Check available monitors
            if len(local_sct.monitors) <= 1:
                print("Warning: No external monitors detected, using primary monitor")
                local_monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            else:
                local_monitor = local_sct.monitors[1]  # Primary monitor
            
            print(f"Screen capture initialized: {local_monitor}")
            
            # Update status to show successful initialization
            self.root.after(0, lambda: self.status_label.config(text="Running: Screen capture ready"))
            
        except Exception as e:
            print(f"Failed to initialize screen capture: {e}")
            self.root.after(0, lambda: self.status_label.config(text="Error: Screen capture failed"))
            self.running = False
            return
        
        while self.running:
            try:
                # Capture screen
                screenshot = local_sct.grab(local_monitor)
                img = np.array(screenshot)
                
                # Convert from BGRA to BGR (remove alpha channel)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Detect cards and end template
                detected_labels, end_detected = self.detector.detect_cards_in_image(img)
                
                # Update detector state
                self.detector.update_detected_cards(detected_labels, end_detected)
                
                # Show end detection in UI
                if end_detected:
                    self.root.after(0, lambda: self.status_label.config(text="End template detected - Memory reset!"))
                    time.sleep(1)  # Brief pause to show message
                
                # Update UI in main thread
                self.root.after(0, self._update_card_display)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Detection error: {e}")
                # Update status to show error
                self.root.after(0, lambda: self.status_label.config(text="Warning: Detection issues"))
                time.sleep(1.0)  # Longer delay on error
        
        # Clean up mss instance
        if local_sct:
            try:
                local_sct.close()
                print("Screen capture cleaned up")
            except:
                pass

    def _update_card_display(self):
        """Update the card display based on detected cards memory"""
        current_detected = self.detector.get_detected_cards()
        memory_detected = self.detector.get_detected_cards_memory()
        
        for card_key, (label, original_color) in self.card_labels.items():
            if card_key in memory_detected:
                # Card in memory - dim it with darker background and faded text
                if card_key in current_detected:
                    # Currently detected - use brighter dim
                    label.config(
                        bg="#303030",  # Lighter gray for currently detected
                        fg="#F9F9F9",  # Brighter gray text
                        relief='sunken',
                        borderwidth=3
                    )
                else:
                    # In memory but not currently detected - darker
                    label.config(
                        bg='#303030',  # Darker gray background
                        fg='#707070',  # Darker gray text
                        relief='sunken',
                        borderwidth=2
                    )
            else:
                # Not in memory - bright with original colors
                label.config(
                    bg='#F8F8FF',  # Light background
                    fg=original_color,  # Original suit color
                    relief='raised',
                    borderwidth=2
                )
        
        # Update status with memory information
        num_current = len(current_detected)
        num_memory = len(memory_detected)
        memory_percentage = (num_memory / 52) * 100
        
        if self.running:
            status_text = f"{num_memory}/52"
        else:
            status_text = f"{num_memory}/52"
            
        self.status_label.config(text=status_text)

    def run(self):
        """Start the UI main loop"""
        try:
            self.root.mainloop()
        finally:
            self.stop_detection()
            self.detector.executor.shutdown(wait=True)

def main():
    # Configuration - Automatically detect template path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "templates")
    threshold = 0.85  # Lowered threshold for better real-time detection
    
    # Verify template path exists
    if not os.path.exists(template_path):
        print(f"ERROR: Template directory not found: {template_path}")
        print("Please ensure 'templates' folder with 52 card templates exists in the same directory.")
        return
    
    print("Initializing card detector...")
    detector = RealtimeCardDetector(template_path, threshold)
    
    print("Starting UI...")
    ui = CardUI(detector)
    ui.run()

if __name__ == "__main__":
    main()