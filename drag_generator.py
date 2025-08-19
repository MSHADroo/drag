import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea, QSizePolicy, QFileDialog, QMessageBox, QFrame, QButtonGroup # Added QButtonGroup
)
from PyQt5.QtGui import QPixmap, QMouseEvent, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QSize, QPoint, pyqtSignal, QRect

# Supported image file extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# --- Modified Class: SingleImageDisplayLabel ---
class SingleImageDisplayLabel(QLabel):
    """
    A custom QLabel for displaying a single image and allowing the user
    to add/move/remove points on it.
    """
    points_changed = pyqtSignal(list) # Signal to notify when points change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self._original_pixmap = QPixmap()
        
        self.current_points = []  # List of QPoint objects
        self.dragging_point_index = -1  # Index of the point being dragged
        self.point_mode_enabled = False # Is point interaction mode active?

        # Store the actual rectangle where the pixmap is drawn within the label
        self._drawn_pixmap_rect = QRect()

    def set_image(self, image_path):
        """Loads and displays an image from the given path. Resets points."""
        self.current_points = [] # Clear points for new image
        if image_path and os.path.exists(image_path):
            self._original_pixmap = QPixmap(image_path)
            self.update_display()
        else:
            self._original_pixmap = QPixmap()
            self.clear() # Clear the label
        self.update() # Request a repaint
        self.points_changed.emit(self.current_points) # Emit to clear points display

    def update_display(self):
        """Scales and sets the pixmap to fit the label and updates _drawn_pixmap_rect."""
        if not self._original_pixmap.isNull():
            label_rect = self.contentsRect()
            scaled_pixmap_size = self._original_pixmap.size().scaled(label_rect.size(), Qt.KeepAspectRatio)
            
            # Calculate the actual rectangle where the scaled pixmap is drawn within the QLabel
            self._drawn_pixmap_rect = QRect(QPoint(0, 0), scaled_pixmap_size)
            self._drawn_pixmap_rect.moveCenter(label_rect.center())
            
            self.setPixmap(self._original_pixmap.scaled(
                label_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.clear()
            self._drawn_pixmap_rect = QRect()

    def resizeEvent(self, event):
        """Recalculates pixmap scaling on resize."""
        self.update_display()
        super().resizeEvent(event)

    def get_original_point_from_display(self, display_point):
        """
        Converts a point from QLabel's display coordinates to the original image's coordinates.
        """
        if self._original_pixmap.isNull() or self._drawn_pixmap_rect.isNull():
            return None
        
        if not self._drawn_pixmap_rect.contains(display_point):
            return None

        x_on_scaled = display_point.x() - self._drawn_pixmap_rect.x()
        y_on_scaled = display_point.y() - self._drawn_pixmap_rect.y()

        scale_x = self._original_pixmap.width() / self._drawn_pixmap_rect.width()
        scale_y = self._original_pixmap.height() / self._drawn_pixmap_rect.height()

        original_x = int(x_on_scaled * scale_x)
        original_y = int(y_on_scaled * scale_y)

        original_x = max(0, min(original_x, self._original_pixmap.width() - 1))
        original_y = max(0, min(original_y, self._original_pixmap.height() - 1))

        return QPoint(original_x, original_y)

    def get_display_point_from_original(self, original_point):
        """
        Converts a point from the original image's coordinates to QLabel's display coordinates.
        """
        if self._original_pixmap.isNull() or self._drawn_pixmap_rect.isNull():
            return None

        scale_x = self._drawn_pixmap_rect.width() / self._original_pixmap.width()
        scale_y = self._drawn_pixmap_rect.height() / self._original_pixmap.height()
        
        x_scaled = int(original_point.x() * scale_x)
        y_scaled = int(original_point.y() * scale_y)
        
        x_display = self._drawn_pixmap_rect.x() + x_scaled
        y_display = self._drawn_pixmap_rect.y() + y_scaled
        
        return QPoint(x_display, y_display)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.point_mode_enabled or self._original_pixmap.isNull():
            return

        original_pos = self.get_original_point_from_display(event.pos())
        if not original_pos:
            return

        if event.button() == Qt.LeftButton:
            self.dragging_point_index = -1
            # Check if an existing point is clicked
            for i, p_orig in enumerate(self.current_points):
                p_display = self.get_display_point_from_original(p_orig)
                if p_display and (event.pos() - p_display).manhattanLength() < 15: # 15 pixel radius
                    self.dragging_point_index = i
                    break
            
            if self.dragging_point_index == -1: # No existing point clicked, add a new one
                self.current_points.append(original_pos)
            self.points_changed.emit(self.current_points)
            self.update() # Request a repaint

        elif event.button() == Qt.RightButton:
            # Remove point on right click
            for i, p_orig in enumerate(self.current_points):
                p_display = self.get_display_point_from_original(p_orig)
                if p_display and (event.pos() - p_display).manhattanLength() < 15:
                    self.current_points.pop(i)
                    self.points_changed.emit(self.current_points)
                    self.update()
                    break

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.point_mode_enabled or self._original_pixmap.isNull():
            return
        
        if self.dragging_point_index != -1 and event.buttons() & Qt.LeftButton:
            new_original_pos = self.get_original_point_from_display(event.pos())
            if new_original_pos:
                self.current_points[self.dragging_point_index] = new_original_pos
                self.points_changed.emit(self.current_points)
                self.update()
        else:
            # Change cursor based on hover
            on_point = False
            if self.point_mode_enabled and not self._original_pixmap.isNull():
                for p_orig in self.current_points:
                    p_display = self.get_display_point_from_original(p_orig)
                    if p_display and (event.pos() - p_display).manhattanLength() < 15:
                        on_point = True
                        break
            if on_point:
                self.setCursor(Qt.OpenHandCursor)
            elif self.point_mode_enabled:
                self.setCursor(Qt.CrossCursor) # Add point mode cursor
            else:
                self.setCursor(Qt.ArrowCursor) # Default cursor

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.point_mode_enabled:
            return
        self.dragging_point_index = -1
        self.setCursor(Qt.ArrowCursor)
        self.points_changed.emit(self.current_points) # Final emit after release

    def paintEvent(self, event):
        super().paintEvent(event)  # Let QLabel draw the pixmap first
        
        if self.point_mode_enabled and not self._original_pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw the points
            painter.setBrush(QBrush(QColor(255, 0, 0))) # Red fill
            painter.setPen(QPen(QColor(0, 0, 0), 1)) # Black border
            for i, p_orig in enumerate(self.current_points):
                p_display = self.get_display_point_from_original(p_orig)
                if p_display:
                    painter.drawEllipse(p_display.x() - 5, p_display.y() - 5, 10, 10) # Small circle for the point
                    painter.drawText(p_display.x() + 8, p_display.y() + 5, str(i + 1)) # Point number

            painter.end()

    def set_point_mode_enabled(self, enabled):
        """Enables or disables point drawing/interaction mode."""
        self.point_mode_enabled = enabled
        if not enabled:
            self.setCursor(Qt.ArrowCursor) # Revert to default cursor
        self.update() # Request a repaint to show/hide points

    def reset_points(self):
        """Clears all points."""
        self.current_points = []
        self.update()
        self.points_changed.emit(self.current_points)

    def get_points_original(self):
        """Returns the current points in original image coordinates."""
        return self.current_points

# --- CombinedImageLabel (No changes from previous version, handles mask) ---
# This class remains mostly the same, it only needs to be used correctly in the UI.
class CombinedImageLabel(QLabel):
    """
    A custom QLabel for displaying a combined (50% transparent overlay) of two images,
    and allowing the user to draw and manipulate a four-point mask (quadrilateral) on it.
    """
    mask_points_changed = pyqtSignal(list) # Signal to notify when mask coordinates change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)

        self._pixmap1 = QPixmap()  # Stores the first original image pixmap
        self._pixmap2 = QPixmap()  # Stores the second original image pixmap
        self.combined_pixmap = QPixmap() # Stores the generated 50/50 combined pixmap
        
        self.mask_points = []  # List of QPoint objects for the 4 mask corners
        self.dragging_point_index = -1  # Index of the mask point being dragged
        
        self.mask_mode_enabled = False  # Is mask interaction mode active?

        # Store the actual rectangle where the pixmap is drawn within the label
        self._drawn_pixmap_rect = QRect()

    def set_images(self, image_path1, image_path2):
        """
        Loads and combines two images. Resets mask points.
        """
        self.mask_points = []  # Reset mask points for new images
        self._pixmap1 = QPixmap(image_path1) if image_path1 else QPixmap()
        self._pixmap2 = QPixmap(image_path2) if image_path2 else QPixmap()
        self._combine_pixmaps()
        self.update()  # Request a repaint

    def _combine_pixmaps(self):
        """
        Combines _pixmap1 and _pixmap2 into combined_pixmap with 50% transparency.
        Assumes both input pixmaps are of the same desired size (e.g., 1024x1024).
        """
        if self._pixmap1.isNull() or self._pixmap2.isNull():
            self.combined_pixmap = QPixmap()
            self.setPixmap(self.combined_pixmap) # Clear the display
            self._drawn_pixmap_rect = QRect() # Reset drawn rect
            return

        # Assuming both images are 1024x1024, or at least have compatible sizes
        # If sizes differ, you might need to decide on a common target size or cropping strategy
        target_size = self._pixmap1.size() # Use size of first pixmap as target

        self.combined_pixmap = QPixmap(target_size)
        self.combined_pixmap.fill(Qt.transparent) # Start with a transparent background

        painter = QPainter(self.combined_pixmap)
        painter.setOpacity(0.5)  # 50% transparency
        painter.drawPixmap(0, 0, self._pixmap1) # Draw original pixmap (no scaling here)
        painter.drawPixmap(0, 0, self._pixmap2) # Draw original pixmap (no scaling here)
        painter.end()

        # Update the QLabel's pixmap to display the combined image
        self.update_display() # Call a new method to handle scaling for display

    def update_display(self):
        """Scales the combined_pixmap to fit the label and updates _drawn_pixmap_rect."""
        if not self.combined_pixmap.isNull():
            label_rect = self.contentsRect()
            scaled_pixmap_size = self.combined_pixmap.size().scaled(label_rect.size(), Qt.KeepAspectRatio)
            
            # Calculate the actual rectangle where the scaled pixmap is drawn within the QLabel
            # This accounts for the alignment (Qt.AlignCenter)
            self._drawn_pixmap_rect = QRect(QPoint(0, 0), scaled_pixmap_size)
            self._drawn_pixmap_rect.moveCenter(label_rect.center())
            
            # Set the pixmap to the scaled version
            self.setPixmap(self.combined_pixmap.scaled(
                label_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.clear()
            self._drawn_pixmap_rect = QRect() # Clear drawn rect

    def resizeEvent(self, event):
        """Recalculates pixmap scaling on resize."""
        self.update_display()
        super().resizeEvent(event)

    def get_original_point_from_display(self, display_point):
        """
        Converts a point from QLabel's display coordinates to the original combined image's coordinates.
        """
        if self.combined_pixmap.isNull() or self._drawn_pixmap_rect.isNull():
            return None

        # Check if the display_point is within the drawn image rectangle
        if not self._drawn_pixmap_rect.contains(display_point):
            return None

        # Convert point from label coordinates to coordinates within the scaled pixmap
        x_on_scaled = display_point.x() - self._drawn_pixmap_rect.x()
        y_on_scaled = display_point.y() - self._drawn_pixmap_rect.y()

        # Calculate scale factors
        scale_x = self.combined_pixmap.width() / self._drawn_pixmap_rect.width()
        scale_y = self.combined_pixmap.height() / self._drawn_pixmap_rect.height()

        # Convert point from scaled pixmap coordinates to original combined pixmap coordinates
        original_x = int(x_on_scaled * scale_x)
        original_y = int(y_on_scaled * scale_y)

        # Ensure points are within the original image bounds
        original_x = max(0, min(original_x, self.combined_pixmap.width() - 1))
        original_y = max(0, min(original_y, self.combined_pixmap.height() - 1))

        return QPoint(original_x, original_y)

    def get_display_point_from_original(self, original_point):
        """
        Converts a point from the original combined image's coordinates to QLabel's display coordinates.
        """
        if self.combined_pixmap.isNull() or self._drawn_pixmap_rect.isNull():
            return None

        # Calculate scale factors
        scale_x = self._drawn_pixmap_rect.width() / self.combined_pixmap.width()
        scale_y = self._drawn_pixmap_rect.height() / self.combined_pixmap.height()
        
        # Convert point from original pixmap coordinates to scaled pixmap coordinates
        x_scaled = int(original_point.x() * scale_x)
        y_scaled = int(original_point.y() * scale_y)
        
        # Convert point from scaled pixmap coordinates to QLabel display coordinates
        x_display = self._drawn_pixmap_rect.x() + x_scaled
        y_display = self._drawn_pixmap_rect.y() + y_scaled
        
        return QPoint(x_display, y_display)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.mask_mode_enabled or self.combined_pixmap.isNull():
            return

        original_pos = self.get_original_point_from_display(event.pos())
        if not original_pos:
            return

        if event.button() == Qt.LeftButton:
            self.dragging_point_index = -1
            # Check if a mask point is clicked
            for i, p_orig in enumerate(self.mask_points):
                p_display = self.get_display_point_from_original(p_orig)
                if p_display and (event.pos() - p_display).manhattanLength() < 15: # 15 pixel radius for clicking on a point
                    self.dragging_point_index = i
                    break
            
            if self.dragging_point_index != -1:
                self.setCursor(Qt.ClosedHandCursor)
            elif len(self.mask_points) < 4:
                # Add a new point if less than 4 points exist
                self.mask_points.append(original_pos)
                if len(self.mask_points) == 4:
                    self.mask_points_changed.emit(self.mask_points) # Emit signal when 4 points are set
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.mask_mode_enabled or self.combined_pixmap.isNull():
            return
        
        if self.dragging_point_index != -1 and event.buttons() & Qt.LeftButton:
            new_original_pos = self.get_original_point_from_display(event.pos())
            if new_original_pos:
                self.mask_points[self.dragging_point_index] = new_original_pos
                self.mask_points_changed.emit(self.mask_points) # Emit signal during drag
                self.update()
        else:
            # Change cursor based on hover
            on_point = False
            if self.mask_mode_enabled and not self.combined_pixmap.isNull():
                for p_orig in self.mask_points:
                    p_display = self.get_display_point_from_original(p_orig)
                    if p_display and (event.pos() - p_display).manhattanLength() < 15:
                        on_point = True
                        break
            if on_point:
                self.setCursor(Qt.OpenHandCursor)
            elif self.mask_mode_enabled:
                self.setCursor(Qt.CrossCursor) # Add point mode cursor
            else:
                self.setCursor(Qt.ArrowCursor) # Default cursor

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.mask_mode_enabled:
            return
        self.dragging_point_index = -1
        self.setCursor(Qt.ArrowCursor)
        self.mask_points_changed.emit(self.mask_points) # Final emit after release

    def paintEvent(self, event):
        super().paintEvent(event)  # Let QLabel draw the combined pixmap first
        
        if self.mask_mode_enabled and not self.combined_pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw the quadrilateral mask
            if len(self.mask_points) > 1:
                painter.setPen(QPen(QColor(0, 255, 0), 2)) # Green line, 2px thickness
                display_points = [self.get_display_point_from_original(p) for p in self.mask_points if self.get_display_point_from_original(p)]
                
                if len(display_points) > 1:
                    # Draw lines connecting points
                    for i in range(len(display_points) - 1):
                        painter.drawLine(display_points[i], display_points[i+1])
                    
                    # Close the quadrilateral if all 4 points are set
                    if len(display_points) == 4:
                        painter.drawLine(display_points[3], display_points[0])

            # Draw the points (handles)
            painter.setBrush(QBrush(QColor(255, 255, 0))) # Yellow fill
            painter.setPen(QPen(QColor(0, 0, 0), 1)) # Black border
            for p_orig in self.mask_points:
                p_display = self.get_display_point_from_original(p_orig)
                if p_display: # Only draw if conversion was successful
                    painter.drawEllipse(p_display.x() - 7, p_display.y() - 7, 14, 14) # A small circle for the handle

            painter.end()

    def set_mask_mode_enabled(self, enabled):
        """Enables or disables mask drawing/interaction mode."""
        self.mask_mode_enabled = enabled
        if not enabled:
            self.setCursor(Qt.ArrowCursor) # Revert to default cursor
        self.update() # Request a repaint to show/hide mask elements

    def reset_mask(self):
        """Clears all mask points."""
        self.mask_points = []
        self.update()
        self.mask_points_changed.emit(self.mask_points) # Notify that mask is reset

    def get_mask_points_original(self):
        """Returns the current mask points in original image coordinates."""
        return self.mask_points

# --- ImageThumbnailWidget (No changes) ---
class ImageThumbnailWidget(QWidget):
    """
    A widget to display an image thumbnail and its folder name.
    Emits a signal when clicked.
    """
    folder_selected = pyqtSignal(str) 

    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.image_paths = self._get_image_paths(folder_path)
        self.selected_thumbnail_index = 0
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(120, 90) # Fixed size for consistency
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("border: 1px solid lightgray;")
        layout.addWidget(self.thumbnail_label)

        self.folder_name_label = QLabel(os.path.basename(folder_path))
        self.folder_name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.folder_name_label)
        
        self.setLayout(layout)
        self.setCursor(Qt.PointingHandCursor)

        self.load_thumbnail()

    def _get_image_paths(self, folder_path):
        """Returns a list of image paths within a folder."""
        image_paths = []
        for file_name in os.listdir(folder_path):
            if any(file_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(folder_path, file_name))
        return sorted(image_paths) # Ensures consistent order

    def load_thumbnail(self):
        """Loads and displays the first image as a thumbnail."""
        if self.image_paths:
            pixmap = QPixmap(self.image_paths[self.selected_thumbnail_index])
            if not pixmap.isNull():
                self.thumbnail_label.setPixmap(pixmap.scaled(self.thumbnail_label.size(),
                                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event: QMouseEvent):
        """Emits the folder_selected signal when clicked."""
        if event.button() == Qt.LeftButton:
            self.folder_selected.emit(self.folder_path)

# --- ImageSelectionWindow (Modified) ---
class ImageSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Point Selector (with Masking)")
        
        # Get primary screen dimensions
        screen_rect = QApplication.primaryScreen().geometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()

        # Set initial window size as a percentage of screen size
        initial_width = int(screen_width * 0.8)
        initial_height = int(screen_height * 0.9)
        self.setGeometry(100, 100, initial_width, initial_height)

        # Set maximum window size to prevent it from going off-screen
        max_width = screen_width - 50 
        max_height = screen_height - 50
        self.setMaximumSize(max_width, max_height)

        self.current_selected_folder = None
        self.image_files_in_folder = []
        self.current_image_index = 0

        self.image_path_frame1 = None  
        self.image_path_frame2 = None  

        self.mask_points_on_combined = [] # Stores the 4 mask points in original image coordinates
        self.source_points_on_frame1 = [] # Stores points for frame 1
        self.target_points_on_frame2 = [] # Stores points for frame 2

        self.current_display_mode = "single_image" # "single_image", "combined_mask"
        self.current_point_selection_target = None # "frame1_points", "frame2_points"

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Directory Selection & Folder List ---
        left_panel = QVBoxLayout()
        
        select_dir_button = QPushButton("انتخاب دایرکتوری")
        select_dir_button.clicked.connect(self.select_directory)
        left_panel.addWidget(select_dir_button)

        self.folder_list_scroll_area = QScrollArea()
        self.folder_list_scroll_area.setWidgetResizable(True)
        self.folder_list_widget_container = QWidget()
        self.folder_list_layout = QVBoxLayout(self.folder_list_widget_container)
        self.folder_list_layout.setAlignment(Qt.AlignTop)
        self.folder_list_scroll_area.setWidget(self.folder_list_widget_container)
        left_panel.addWidget(self.folder_list_scroll_area)

        main_layout.addLayout(left_panel, 1)

        # --- Center Panel: Thumbnails & Controls ---
        center_panel = QVBoxLayout()
        
        self.thumbnail_grid_scroll_area = QScrollArea()
        self.thumbnail_grid_scroll_area.setWidgetResizable(True)
        self.thumbnail_grid_widget_container = QWidget()
        
        self.thumbnail_grid_layout = QHBoxLayout(self.thumbnail_grid_widget_container)
        self.thumbnail_grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.thumbnail_grid_scroll_area.setWidget(self.thumbnail_grid_widget_container)
        
        self.thumbnail_grid_scroll_area.setMinimumWidth(350) 
        center_panel.addWidget(self.thumbnail_grid_scroll_area, 1) 

        nav_buttons_layout = QHBoxLayout()
        self.prev_image_button = QPushButton("عکس قبلی")
        self.prev_image_button.clicked.connect(self.show_previous_image)
        self.prev_image_button.setEnabled(False)
        nav_buttons_layout.addWidget(self.prev_image_button)

        self.next_image_button = QPushButton("عکس بعدی")
        self.next_image_button.clicked.connect(self.show_next_image)
        self.next_image_button.setEnabled(False)
        nav_buttons_layout.addWidget(self.next_image_button)
        center_panel.addLayout(nav_buttons_layout)

        frame_buttons_layout = QHBoxLayout()
        self.select_frame1_button = QPushButton("انتخاب برای فریم اول")
        self.select_frame1_button.clicked.connect(self.set_frame1_image)
        self.select_frame1_button.setEnabled(False)
        frame_buttons_layout.addWidget(self.select_frame1_button)

        self.select_frame2_button = QPushButton("انتخاب برای فریم دوم")
        self.select_frame2_button.clicked.connect(self.set_frame2_image)
        self.select_frame2_button.setEnabled(False)
        frame_buttons_layout.addWidget(self.select_frame2_button)
        center_panel.addLayout(frame_buttons_layout)
        
        # Point selection buttons
        point_selection_layout = QHBoxLayout()
        self.toggle_source_points_button = QPushButton("انتخاب نقاط مبدا (فریم ۱)")
        self.toggle_source_points_button.clicked.connect(lambda: self.toggle_point_selection_mode("frame1_points"))
        self.toggle_source_points_button.setEnabled(False)
        point_selection_layout.addWidget(self.toggle_source_points_button)

        self.toggle_target_points_button = QPushButton("انتخاب نقاط مقصد (فریم ۲)")
        self.toggle_target_points_button.clicked.connect(lambda: self.toggle_point_selection_mode("frame2_points"))
        self.toggle_target_points_button.setEnabled(False)
        point_selection_layout.addWidget(self.toggle_target_points_button)
        center_panel.addLayout(point_selection_layout)

        # Mask buttons
        mask_buttons_layout = QHBoxLayout()
        self.toggle_mask_mode_button = QPushButton("فعال کردن حالت ماسک")
        self.toggle_mask_mode_button.clicked.connect(self.toggle_mask_mode)
        self.toggle_mask_mode_button.setEnabled(False) 
        mask_buttons_layout.addWidget(self.toggle_mask_mode_button)

        self.reset_mask_button = QPushButton("ریست ماسک")
        self.reset_mask_button.clicked.connect(self.reset_mask_points)
        self.reset_mask_button.setEnabled(False) 
        mask_buttons_layout.addWidget(self.reset_mask_button)
        center_panel.addLayout(mask_buttons_layout)

        self.save_button = QPushButton("ذخیره مختصات")
        self.save_button.clicked.connect(self.save_coordinates_to_json)
        self.save_button.setEnabled(False)
        center_panel.addWidget(self.save_button)

        main_layout.addLayout(center_panel, 2)

        # --- Right Panel: Image Displays & Coordinates ---
        right_panel = QVBoxLayout()
        
        # Container for the two image labels to manage their visibility
        image_display_container = QFrame() # Using QFrame as a container for switching visibility
        image_display_container_layout = QVBoxLayout(image_display_container)
        image_display_container_layout.setContentsMargins(0, 0, 0, 0) # No extra margins

        # Single Image Display Label
        self.single_image_display_label = SingleImageDisplayLabel()
        self.single_image_display_label.setStyleSheet("border: 1px solid gray;")
        self.single_image_display_label.points_changed.connect(self.update_single_image_points_display) # Connect new signal
        image_display_container_layout.addWidget(self.single_image_display_label)
        
        # Combined Image Display Label (for mask)
        self.combined_image_display_label = CombinedImageLabel()
        self.combined_image_display_label.setStyleSheet("border: 1px solid gray;")
        self.combined_image_display_label.mask_points_changed.connect(self.update_mask_coord_label)
        image_display_container_layout.addWidget(self.combined_image_display_label)

        right_panel.addWidget(image_display_container, 1) # Stretch factor 1

        # Initially, show the single image display and hide combined
        self.single_image_display_label.show()
        self.combined_image_display_label.hide()
        
        # Labels for coordinates
        self.source_points_label = QLabel("نقاط مبدا (فریم ۱): (نقطه ای انتخاب نشده)")
        self.source_points_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.source_points_label)
        
        self.target_points_label = QLabel("نقاط مقصد (فریم ۲): (نقطه ای انتخاب نشده)")
        self.target_points_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.target_points_label)

        self.mask_coord_label = QLabel("مختصات ماسک: (ماسک تعریف نشده)")
        self.mask_coord_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.mask_coord_label)

        # Frame information labels
        frame_info_layout = QHBoxLayout()
        self.frame1_info_label = QLabel("فریم اول: -")
        self.frame2_info_label = QLabel("فریم دوم: -")
        frame_info_layout.addWidget(self.frame1_info_label)
        frame_info_layout.addWidget(self.frame2_info_label)
        right_panel.addLayout(frame_info_layout)

        main_layout.addLayout(right_panel, 4)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "انتخاب دایرکتوری حاوی پوشه تصاویر")
        if directory:
            self.clear_layouts()
            self.scan_directory(directory)
            if not self.folder_list_layout.count():
                QMessageBox.information(self, "اطلاعات", "پوشه‌ای با بیش از یک عکس در این دایرکتوری یافت نشد.")

    def clear_layouts(self):
        # Clear folder list
        while self.folder_list_layout.count():
            item = self.folder_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear thumbnail grid
        while self.thumbnail_grid_layout.count():
            item = self.thumbnail_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset image displays and their states
        self.single_image_display_label.set_image(None)
        self.single_image_display_label.reset_points()
        self.single_image_display_label.set_point_mode_enabled(False) # Disable point mode initially

        self.combined_image_display_label.set_images(None, None)
        self.combined_image_display_label.reset_mask()
        self.combined_image_display_label.set_mask_mode_enabled(False)

        # Ensure only single image display is visible initially
        self.single_image_display_label.show()
        self.combined_image_display_label.hide()

        # Reset coordinate labels
        self.source_points_label.setText("نقاط مبدا (فریم ۱): (نقطه ای انتخاب نشده)")
        self.target_points_label.setText("نقاط مقصد (فریم ۲): (نقطه ای انتخاب نشده)")
        self.mask_coord_label.setText("مختصات ماسک: (ماسک تعریف نشده)")

        # Reset internal data
        self.current_selected_folder = None
        self.image_files_in_folder = []
        self.current_image_index = 0
        self.image_path_frame1 = None
        self.image_path_frame2 = None
        self.mask_points_on_combined = []
        self.source_points_on_frame1 = []
        self.target_points_on_frame2 = []
        self.current_display_mode = "single_image"
        self.current_point_selection_target = None

        # Reset button states
        self.prev_image_button.setEnabled(False)
        self.next_image_button.setEnabled(False)
        self.select_frame1_button.setEnabled(False)
        self.select_frame2_button.setEnabled(False)
        self.toggle_source_points_button.setEnabled(False)
        self.toggle_source_points_button.setText("انتخاب نقاط مبدا (فریم ۱)")
        self.toggle_target_points_button.setEnabled(False)
        self.toggle_target_points_button.setText("انتخاب نقاط مقصد (فریم ۲)")
        self.toggle_mask_mode_button.setEnabled(False)
        self.toggle_mask_mode_button.setText("فعال کردن حالت ماسک") 
        self.reset_mask_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        self.frame1_info_label.setText("فریم اول: -")
        self.frame2_info_label.setText("فریم دوم: -")

    def scan_directory(self, root_dir):
        for entry in os.listdir(root_dir):
            full_path = os.path.join(root_dir, entry)
            if os.path.isdir(full_path):
                image_count = 0
                for file_name in os.listdir(full_path):
                    if any(file_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                        image_count += 1
                
                if image_count > 1:
                    thumbnail_widget = ImageThumbnailWidget(full_path)
                    thumbnail_widget.folder_selected.connect(self.select_folder) 
                    self.folder_list_layout.addWidget(thumbnail_widget)

    def select_folder(self, folder_path):
        self.current_selected_folder = folder_path
        self.image_files_in_folder = []
        for file_name in os.listdir(folder_path):
            if any(file_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                self.image_files_in_folder.append(os.path.join(folder_path, file_name))
        
        self.image_files_in_folder.sort()
        self.current_image_index = 0 # Reset to first image in new folder
        
        while self.thumbnail_grid_layout.count():
            item = self.thumbnail_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, img_path in enumerate(self.image_files_in_folder):
            label = QLabel()
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(100, 75, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setFixedSize(110, 85)
                label.setStyleSheet("border: 1px solid lightblue;")
                label.setAlignment(Qt.AlignCenter)
                # Connect click event to _thumbnail_clicked
                label.mousePressEvent = lambda event, path=img_path, idx=i: self._thumbnail_clicked(event, path, idx)
                self.thumbnail_grid_layout.addWidget(label)
        
        # Display the first image of the selected folder in the single display label
        if self.image_files_in_folder:
            self.display_single_image(self.image_files_in_folder[0])
        else:
            self.single_image_display_label.set_image(None)

        self.update_navigation_buttons()
        self.select_frame1_button.setEnabled(True)
        self.select_frame2_button.setEnabled(True)
        # Enable point selection buttons if frames are selected
        if self.image_path_frame1:
            self.toggle_source_points_button.setEnabled(True)
        if self.image_path_frame2:
            self.toggle_target_points_button.setEnabled(True)
        
        self.check_save_button_status() # Re-evaluate save button status

    def _thumbnail_clicked(self, event, image_path, index):
        """
        Handles click on an individual image thumbnail.
        Displays the clicked image in the single image display label.
        """
        if event.button() == Qt.LeftButton:
            self.current_image_index = index # Update current index
            self.display_single_image(image_path) # Display the clicked image
            self.update_navigation_buttons() # Update prev/next button states

    def display_single_image(self, image_path):
        """Switches to single image display mode and loads the image."""
        self.current_display_mode = "single_image"
        self.combined_image_display_label.hide()
        self.combined_image_display_label.set_mask_mode_enabled(False) # Disable mask mode

        self.single_image_display_label.show()
        self.single_image_display_label.set_image(image_path)
        # Re-enable point mode based on current target if in single image mode
        if self.current_point_selection_target == "frame1_points" and image_path == self.image_path_frame1:
            self.single_image_display_label.set_point_mode_enabled(True)
            self.single_image_display_label.current_points = list(self.source_points_on_frame1) # Load current points
        elif self.current_point_selection_target == "frame2_points" and image_path == self.image_path_frame2:
            self.single_image_display_label.set_point_mode_enabled(True)
            self.single_image_display_label.current_points = list(self.target_points_on_frame2) # Load current points
        else:
            self.single_image_display_label.set_point_mode_enabled(False)
            self.single_image_display_label.reset_points() # Clear points displayed if not in active selection mode

    def display_combined_image(self):
        """
        Updates the CombinedImageLabel with the selected Frame 1 and Frame 2 images.
        Switches visibility to CombinedImageLabel.
        """
        if self.image_path_frame1 and self.image_path_frame2:
            self.current_display_mode = "combined_mask"
            self.single_image_display_label.hide()
            self.single_image_display_label.set_point_mode_enabled(False) # Disable point mode

            self.combined_image_display_label.set_images(self.image_path_frame1, self.image_path_frame2)
            self.combined_image_display_label.show()
            self.toggle_mask_mode_button.setEnabled(True)
            self.reset_mask_button.setEnabled(True)
        else:
            # If not both frames are selected, clear combined display and show single
            self.combined_image_display_label.set_images(None, None)
            self.combined_image_display_label.hide()
            self.combined_image_display_label.set_mask_mode_enabled(False)
            self.single_image_display_label.show() # Fallback to showing single image (if any)
            self.toggle_mask_mode_button.setEnabled(False)
            self.toggle_mask_mode_button.setText("فعال کردن حالت ماسک")
            self.reset_mask_button.setEnabled(False)
        self.check_save_button_status()

    def update_navigation_buttons(self):
        if self.image_files_in_folder:
            self.prev_image_button.setEnabled(self.current_image_index > 0)
            self.next_image_button.setEnabled(self.current_image_index < len(self.image_files_in_folder) - 1)
        else:
            self.prev_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_single_image(self.image_files_in_folder[self.current_image_index])
            self.update_navigation_buttons()

    def show_next_image(self):
        if self.current_image_index < len(self.image_files_in_folder) - 1:
            self.current_image_index += 1
            self.display_single_image(self.image_files_in_folder[self.current_image_index])
            self.update_navigation_buttons()

    def update_single_image_points_display(self, points):
        """
        Updates the coordinate label based on which points are being edited.
        This slot is connected to SingleImageDisplayLabel.points_changed.
        """
        points_str = ""
        if points:
            points_str = ", ".join([f"({p.x()}, {p.y()})" for p in points])
        else:
            points_str = "(نقطه ای انتخاب نشده)"
        
        if self.current_point_selection_target == "frame1_points":
            self.source_points_on_frame1 = list(points) # Store a copy
            self.source_points_label.setText(f"نقاط مبدا (فریم ۱): {points_str}")
        elif self.current_point_selection_target == "frame2_points":
            self.target_points_on_frame2 = list(points) # Store a copy
            self.target_points_label.setText(f"نقاط مقصد (فریم ۲): {points_str}")
        self.check_save_button_status()

    def update_mask_coord_label(self, points):
        self.mask_points_on_combined = points
        points_str = ""
        if points:
            points_str = ", ".join([f"({p.x()}, {p.y()})" for p in points])
        else:
            points_str = "(ماسک تعریف نشده)"
        self.mask_coord_label.setText(f"مختصات ماسک: {points_str}")
        self.check_save_button_status()

    def set_frame1_image(self):
        if self.current_selected_folder and self.image_files_in_folder:
            self.image_path_frame1 = self.image_files_in_folder[self.current_image_index]
            QMessageBox.information(self, "فریم اول", f"تصویر '{os.path.basename(self.image_path_frame1)}' به عنوان فریم اول انتخاب شد.")
            self.update_frame_info_labels()
            # If frame 1 selected, enable its point selection button
            self.toggle_source_points_button.setEnabled(True)
            self.display_single_image(self.image_path_frame1) # Show selected frame 1
            self.check_save_button_status()

    def set_frame2_image(self):
        if self.current_selected_folder and self.image_files_in_folder:
            self.image_path_frame2 = self.image_files_in_folder[self.current_image_index]
            QMessageBox.information(self, "فریم دوم", f"تصویر '{os.path.basename(self.image_path_frame2)}' به عنوان فریم دوم انتخاب شد.")
            self.update_frame_info_labels()
            # If frame 2 selected, enable its point selection button
            self.toggle_target_points_button.setEnabled(True)
            self.display_single_image(self.image_path_frame2) # Show selected frame 2
            self.check_save_button_status()
            
    def update_frame_info_labels(self):
        frame1_name = os.path.basename(self.image_path_frame1) if self.image_path_frame1 else "-"
        frame2_name = os.path.basename(self.image_path_frame2) if self.image_path_frame2 else "-"

        self.frame1_info_label.setText(f"فریم اول: {frame1_name}")
        self.frame2_info_label.setText(f"فریم دوم: {frame2_name}")

        # Enable mask buttons if both frames are selected
        if self.image_path_frame1 and self.image_path_frame2:
            self.toggle_mask_mode_button.setEnabled(True)
            self.reset_mask_button.setEnabled(True)
        else:
            self.toggle_mask_mode_button.setEnabled(False)
            self.reset_mask_button.setEnabled(False)

        self.check_save_button_status()

    def toggle_point_selection_mode(self, target_points):
        """
        Toggles point selection mode for either source or target points.
        target_points can be "frame1_points" or "frame2_points".
        """
        # Disable mask mode first if it's active
        self.combined_image_display_label.set_mask_mode_enabled(False)
        self.toggle_mask_mode_button.setText("فعال کردن حالت ماسک")

        # Determine if we are switching to a new target or disabling current one
        if self.current_point_selection_target == target_points and self.single_image_display_label.point_mode_enabled:
            # If same target and currently enabled, disable it
            self.single_image_display_label.set_point_mode_enabled(False)
            self.current_point_selection_target = None
            self.toggle_source_points_button.setText("انتخاب نقاط مبدا (فریم ۱)")
            self.toggle_target_points_button.setText("انتخاب نقاط مقصد (فریم ۲)")
        else:
            # Activate point selection for the chosen target
            self.current_point_selection_target = target_points
            self.single_image_display_label.set_point_mode_enabled(True)
            
            if target_points == "frame1_points":
                if self.image_path_frame1:
                    self.display_single_image(self.image_path_frame1)
                    self.single_image_display_label.current_points = list(self.source_points_on_frame1) # Load saved points
                    self.single_image_display_label.update()
                    self.toggle_source_points_button.setText("غیرفعال کردن انتخاب نقاط مبدا")
                    self.toggle_target_points_button.setText("انتخاب نقاط مقصد (فریم ۲)") # Reset other button text
                else:
                    QMessageBox.warning(self, "خطا", "لطفاً ابتدا فریم اول را انتخاب کنید.")
                    self.current_point_selection_target = None
                    self.single_image_display_label.set_point_mode_enabled(False)
            elif target_points == "frame2_points":
                if self.image_path_frame2:
                    self.display_single_image(self.image_path_frame2)
                    self.single_image_display_label.current_points = list(self.target_points_on_frame2) # Load saved points
                    self.single_image_display_label.update()
                    self.toggle_target_points_button.setText("غیرفعال کردن انتخاب نقاط مقصد")
                    self.toggle_source_points_button.setText("انتخاب نقاط مبدا (فریم ۱)") # Reset other button text
                else:
                    QMessageBox.warning(self, "خطا", "لطفاً ابتدا فریم دوم را انتخاب کنید.")
                    self.current_point_selection_target = None
                    self.single_image_display_label.set_point_mode_enabled(False)
        
        self.single_image_display_label.update() # Ensure display reflects point mode change
        self.check_save_button_status() # Re-evaluate save button status

    def toggle_mask_mode(self):
        # Disable point selection modes first
        self.single_image_display_label.set_point_mode_enabled(False)
        self.current_point_selection_target = None
        self.toggle_source_points_button.setText("انتخاب نقاط مبدا (فریم ۱)")
        self.toggle_target_points_button.setText("انتخاب نقاط مقصد (فریم ۲)")
        self.single_image_display_label.reset_points() # Clear displayed points from single image

        new_state = not self.combined_image_display_label.mask_mode_enabled
        self.combined_image_display_label.set_mask_mode_enabled(new_state)
        self.toggle_mask_mode_button.setText("غیرفعال کردن حالت ماسک" if new_state else "فعال کردن حالت ماسک")
        
        if new_state: # If mask mode is enabled, switch to combined image display
            self.display_combined_image()
        else: # If mask mode is disabled, revert to showing single image (e.g., current thumbnail)
            if self.image_files_in_folder:
                self.display_single_image(self.image_files_in_folder[self.current_image_index])
            else:
                self.single_image_display_label.set_image(None) # Clear if no image in folder
        self.check_save_button_status()


    def reset_mask_points(self):
        self.combined_image_display_label.reset_mask()
        self.update_mask_coord_label([]) # Update label directly after reset

    def check_save_button_status(self):
        if (self.image_path_frame1 and self.image_path_frame2 and
            len(self.mask_points_on_combined) == 4 and
            len(self.source_points_on_frame1) > 0 and 
            len(self.source_points_on_frame1) == len(self.target_points_on_frame2)):
            self.save_button.setEnabled(True)
        else:
            self.save_button.setEnabled(False)

    def save_coordinates_to_json(self):
        if not self.image_path_frame1 or not self.image_path_frame2:
            QMessageBox.warning(self, "خطا", "لطفاً هر دو فریم را انتخاب کنید.")
            return

        if len(self.mask_points_on_combined) != 4:
            QMessageBox.warning(self, "خطا", "لطفاً دقیقاً 4 نقطه برای ماسک انتخاب کنید.")
            return
        
        if not self.source_points_on_frame1 or not self.target_points_on_frame2:
            QMessageBox.warning(self, "خطا", "لطفاً نقاط مبدا و مقصد را انتخاب کنید.")
            return
        
        if len(self.source_points_on_frame1) != len(self.target_points_on_frame2):
            QMessageBox.warning(self, "خطا", "تعداد نقاط مبدا و مقصد باید برابر باشد.")
            return

        data = {
            "frame1_image": os.path.basename(self.image_path_frame1),
            "frame2_image": os.path.basename(self.image_path_frame2),
            "mask_area": [],
            "source_points": [],
            "target_points": []
        }

        for p in self.mask_points_on_combined:
            data["mask_area"].append({"x": p.x(), "y": p.y()})
        
        for p in self.source_points_on_frame1:
            data["source_points"].append({"x": p.x(), "y": p.y()})

        for p in self.target_points_on_frame2:
            data["target_points"].append({"x": p.x(), "y": p.y()})
        
        output_folder = os.path.dirname(self.image_path_frame1)
        # Use a more descriptive file name, e.g., combining names of frame1 and frame2
        base1 = os.path.splitext(os.path.basename(self.image_path_frame1))[0]
        base2 = os.path.splitext(os.path.basename(self.image_path_frame2))[0]
        output_file_name = f"drag_data_{base1}_to_{base2}.json"
        output_path = os.path.join(output_folder, output_file_name)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "ذخیره شد", f"مختصات با موفقیت در:\n{output_path}\nذخیره شد.")
        except Exception as e:
            QMessageBox.critical(self, "خطا در ذخیره", f"خطا در ذخیره فایل JSON: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSelectionWindow()
    window.show()
    sys.exit(app.exec_())