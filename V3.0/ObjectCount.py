# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(BaseSolution):

    def __init__(self, allowed_classes=None, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class
        self.region_initialized = False  # Bool variable for region initialization
        
        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]

        # ADDED lines 22 through 23 on 20/08/2025 | Choose specific classes or set to None
        self.allowed_classes = allowed_classes  # list of names or None

    def reset_count(self):
        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class
    
    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """
        Counts objects within a polygonal or linear region based on their tracks.

        Args:
            current_centroid (Tuple[float, float]): Current centroid values in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id = 1
            >>> prev_position = (120, 220)
            >>> cls = 0
            >>> counter.count_objects(current_centroid, track_id, prev_position, cls)
        """
        if prev_position is None or track_id in self.counted_ids:
            return
        
        # cls_name = self.names[cls]
        # self.store_classwise_counts(cls)  # ensure safe counts

        if len(self.region) == 2:  # Linear region (defined as a line segment)
            line = self.LineString(self.region)  # Check if the line intersects the trajectory of the object
            if line.intersects(self.LineString([prev_position, current_centroid])):
                # Determine orientation of the region (vertical or horizontal)
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x-coordinates to determine direction
                    if current_centroid[0] > prev_position[0]:  # Moving right
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                    else:  # Moving left
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                # Horizontal region: Compare y-coordinates to determine direction
                elif current_centroid[1] > prev_position[1]:  # Moving downward
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:  # Moving upward
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # Polygonal region
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                # Determine motion direction for vertical or horizontal polygons
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (
                    region_width < region_height
                    and current_centroid[0] > prev_position[0]
                    or region_width >= region_height
                    and current_centroid[1] > prev_position[1]
                ):  # Moving right
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:  # Moving left
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates.

        This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
        initializing 'IN' and 'OUT' counts to zero if the class is not already present.

        Examples:
            >>> counter = ObjectCounter()
            >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
            >>> print(counter.classwise_counts)
            {'person': {'IN': 0, 'OUT': 0}}
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        # ADDED lines 125 through 141 on 21/08/2025
        # Determine which classes to display
        if self.allowed_classes:
            display_classes = self.allowed_classes
        else:
            display_classes = list(self.names.values())  # all classes from model

        labels_dict = {}
        for class_name in display_classes:
            # Ensure storage exists for the class
            if class_name not in self.classwise_counts:
                self.classwise_counts[class_name] = {"IN": 0, "OUT": 0}

            labels_dict[str.capitalize(class_name)] = (
                f"{'IN ' + str(self.classwise_counts[class_name]['IN']) if self.show_in else ''} "
                f"{'OUT ' + str(self.classwise_counts[class_name]['OUT']) if self.show_out else ''}".strip()
            ) # REPLACEMENT for lines 143 through 148

        # labels_dict = {
        #     str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
        #     f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
        #     for key, value in self.classwise_counts.items()
        #     if value["IN"] != 0 or value["OUT"] != 0
        # }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        """
        Processes input data (frames or object tracks) and updates object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
        object counts, and displays the results on the input image.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.

        Returns:
            (numpy.ndarray): The processed image with annotations and count information.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = counter.count(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # ADDED lines 162 through 165 on 20/08/2025 | Check against allowed classes
            class_name = self.names[cls]
            if self.allowed_classes and class_name not in self.allowed_classes:
                continue

            self.store_tracking_history(track_id, box)  # Store track history
            self.store_classwise_counts(cls)  # store classwise counts in dict

            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            # store previous position of track for object counting
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(current_centroid, track_id, prev_position, cls)  # Perform object counting

        return im0  # return output image for more usage