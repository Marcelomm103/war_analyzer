import cv2
import numpy as np
from typing import Tuple, List
import random
import pytesseract

current_game_state = None

class Territory:
    def __init__(self, name: str, continent: str, number: int, armies: int, player: str, position: Tuple[int, int]):
        self.name = name
        self.continent = continent
        self.number = number
        self.armies = armies
        self.player = player
        self.position = position


class Player:
    def __init__(self, name: str, player_color: str):
        self.name = name
        self.player_color = player_color
        self.num_armies = 0
        self.num_territories = 0
        self.territories = []
        self.objective = None  # Will be assigned an Objective instance later
        self.bonus_armies = 0
        self.cards: List['Card'] = []  # Max 5 cards


class Objective:
    def __init__(self, obj_number: int, obj_description: str):
        self.obj_number = obj_number
        self.completed = False
        self.obj_changed = False
        self.obj_description = obj_description


class Card:
    def __init__(self, symbol: str, symbol_number: int, backup_armies: int, territory_number: int):
        self.symbol = symbol
        self.symbol_number = symbol_number
        self.backup_armies = backup_armies
        self.territory_number = territory_number

def initialize_players(num_players: int) -> List[Player]:
    """
    Initializes the player instances with random unique colors.

    Args:
        num_players (int): Number of players (minimum 3).

    Returns:
        List[Player]: A list of initialized Player instances.
    """
    if num_players < 3 or num_players > 6:
        raise ValueError("Number of players must be between 3 and 6.")

    available_colors = ["blue", "red", "yellow", "green", "gray", "purple"]
    selected_colors = random.sample(available_colors, num_players)

    players = [Player(name="", player_color=color) for color in selected_colors]

    return players

def initialize_objectives():
    """Creates and returns a list of Objective instances."""
    objectives = [
        Objective(1, "Conquer 24 territories."),
        Objective(2, "Occupy 18 territories with at least 2 armies each."),
        Objective(3, "Conquer Asía and África."),
        Objective(4, "Conquer North América and África."),
        Objective(5, "Conquer North América and Oceania"),
        Objective(6, "Conquer Asía and South America"),
        Objective(7, "Conquer Europe, South America and a third continent"),
        Objective(8, "Conquer Europe, North America and a third continent"),
        Objective(9, "Destroy all the green armies. If you're this color army, or anyone else destroy it, your objective is now conquer 24 territories."),
        Objective(10, "Destroy all the yellow armies. If you're this color army, or anyone else destroy it, your objective is now conquer 24 territories."),
        Objective(11, "Destroy all the purple armies. If you're this color army, or anyone else destroy it, your objective is now conquer 24 territories."),
        Objective(12, "Destroy all the gray armies. If you're this color army, or anyone else destroy it, your objective is now conquer 24 territories."),
        Objective(13, "Destroy all the red armies. If you're this color army, or anyone else destroy it, your objective is now conquer 24 territories."),
        Objective(14, "Destroy all the blue armies. If you're this color army, or anyone else destroy it, your objective is now conquer 24 territories.")
    ]
    return objectives

def get_objectives(players):
    """Randomly assigns an objective to each player."""
    objectives = initialize_objectives()
    random.shuffle(objectives)  # Shuffle to ensure randomness

    for player in players:
        player.objective = objectives.pop()  # Assign and remove from the list

def extract_game_state(frame, territories, players):
    game_state = {}
    for territory in territories:
        troops = detect_troop_count(frame, territory)
        owner = detect_owner_rgb(frame, territory, players)
        game_state[territory.name] = {
            "owner": owner,
            "troops": troops,
            "position": territory.position,
            "continent": territory.continent
        }
    return game_state

def print_game_state(game_state, players):
    """Prints the current game state in a human-readable format."""
    # Initialize with all player colors + 'neutral'
    player_colors = [player.player_color for player in players]
    player_territories = {color: [] for color in player_colors}
    player_territories["neutral"] = []  # For unclaimed territories
    
    for territory_name, data in game_state.items():
        owner = data["owner"].lower()  # Normalize case
        
        # Handle unknown owners
        if owner not in player_territories:
            owner = "neutral"
            
        player_territories[owner].append(
            (territory_name, data["troops"], data["continent"])
        )
    
    # Print player territories
    for player in players:
        color = player.player_color
        territories = player_territories[color]
        
        print(f"\nPlayer {player.name} ({color}) - Territories: {len(territories)}")
        print("-" * 40)
        
        # Group by continent
        continents = {}
        for terr_name, troops, continent in territories:
            if continent not in continents:
                continents[continent] = []
            continents[continent].append(f"{terr_name} ({troops})")
        
        for continent, terr_list in continents.items():
            print(f"  {continent}:")
            print("    " + ", ".join(terr_list))
    
    # Print neutral territories
    neutral_terr = player_territories["neutral"]
    if neutral_terr:
        print("\nNeutral/Unknown Territories:")
        print("-" * 40)
        for terr_name, troops, continent in neutral_terr:
            print(f"  {terr_name} ({troops}) - {continent}")

def detect_owner(frame, territory, players, debug=False):
    """
    Improved ownership detection with:
    - Region sampling
    - HSV range validation
    - Majority voting
    - Debug visualization
    """
    x, y = territory.position
    sample_radius = 5  # Pixels around center to sample
    min_color_match = 0.5  # 60% of pixels must match
    
    # Sample a small region
    region = frame[y-sample_radius:y+sample_radius, 
                  x-sample_radius:x+sample_radius]
    if region.size == 0:
        return "neutral"
    
    # Convert to HSV
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Check against all player colors
    match_counts = {}
    for player in players:
        lower, upper = get_hsv_bounds(player.player_color)
        mask = cv2.inRange(hsv_region, lower, upper)
        match_counts[player.player_color] = np.count_nonzero(mask)
    
    # Find best match
    total_pixels = region.shape[0] * region.shape[1]
    best_match = max(match_counts.items(), key=lambda x: x[1])
    
    if debug:
        debug_visualization(frame, territory, hsv_region, match_counts)
    
    # Verify minimum match threshold
    if best_match[1] / total_pixels >= min_color_match:
        return best_match[0]
    return "neutral"

def detect_owner_advanced(frame, position, players):
    x,y = position
    sample = frame[y-5:y+5, x-5:x+5]  # 10x10 sample area
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    
    best_match = None
    highest_score = 0
    
    for player in players:
        # Check in both color spaces
        hsv_mask = cv2.inRange(hsv, *get_hsv_bounds(player.color))
        lab_mask = cv2.inRange(lab, *get_lab_bounds(player.color))
        
        combined = cv2.bitwise_and(hsv_mask, lab_mask)
        score = np.sum(combined)/255  # Count matching pixels
        
        if score > highest_score:
            highest_score = score
            best_match = player.color
    
    return best_match if highest_score > 25 else "neutral"  # Minimum 25 matching pixels

def detect_owner_rgb(frame, territory, players):
    """Detect owner using RGB color ranges while ignoring white pixels"""
    x, y = territory.position
    sample_radius = 10  # Pixels to sample around center
    min_match_percent = 0.3  # 70% of non-white pixels must match
    white_threshold = 230  # BGR threshold for white pixels
    
    # Sample a region around the territory center
    region = frame[y-sample_radius:y+sample_radius, 
                  x-sample_radius:x+sample_radius]
    
    if region.size == 0:
        return "neutral"
    
    # Create mask for non-white pixels
    white_mask = cv2.inRange(region, 
                            (white_threshold, white_threshold, white_threshold),
                            (255, 255, 255))
    non_white_pixels = cv2.bitwise_not(white_mask)
    total_non_white = cv2.countNonZero(non_white_pixels)
    
    # If all pixels are white, return neutral
    if total_non_white == 0:
        return "neutral"
    
    # Define RGB ranges for each player (BGR format - OpenCV convention)
    rgb_ranges = {
        "red": ((22, 0, 186), (32, 100, 196)),
        "blue": ((178, 159, 0), (188, 169, 0)),
        "green": ((45, 165, 26), (55, 175, 36)),
        "yellow": ((0, 190, 213), (0, 200, 223)),
        "purple": ((110, 0, 82), (120, 0, 92)),
        "gray": ((51, 51, 51), (61, 61, 61))
    }
    
    # Count matching non-white pixels for each player
    match_counts = {}
    for player in players:
        lower = np.array(rgb_ranges[player.player_color][0], dtype="uint8")
        upper = np.array(rgb_ranges[player.player_color][1], dtype="uint8")
        
        # Get color matches
        color_mask = cv2.inRange(region, lower, upper)
        
        # Combine with non-white mask
        final_mask = cv2.bitwise_and(color_mask, non_white_pixels)
        match_counts[player.player_color] = cv2.countNonZero(final_mask)
    
    # Determine best match
    best_match = max(match_counts.items(), key=lambda x: x[1])
    
    # Apply confidence threshold based on non-white pixels
    if (best_match[1] / total_non_white) >= min_match_percent:
        return best_match[0]
    return "neutral"

def get_hsv_bounds(color):
    """Returns HSV bounds for each player color using calibrated values"""
    hsv_ranges = {
        "red": ((167, 245, 183), (177, 255, 193)),
        "blue": ((86, 245, 202), (96, 255, 212)),
        "green": ((44, 40, 195), (54, 255, 205)),
        "yellow": ((22, 245, 245), (32, 255, 255)),
        "purple": ((130, 184, 122), (140, 194, 132)),
        "gray": ((0, 0, 61), (10, 10, 71))
    }
    return hsv_ranges.get(color.lower(), ((0, 0, 0), (180, 255, 255)))

def debug_visualization(frame, territory, hsv_region, match_counts):
    """Shows detection details for troubleshooting"""
    x, y = territory.position
    # Mark territory center
    cv2.drawMarker(frame, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    
    # Show HSV sample region
    sample_display = cv2.resize(hsv_region, (200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("HSV Sample", cv2.cvtColor(sample_display, cv2.COLOR_HSV2BGR))
    
    # Print matching results
    print(f"\n--- {territory.name} Detection Report ---")
    for color, count in match_counts.items():
        percentage = (count / (hsv_region.shape[0] * hsv_region.shape[1])) * 100
        print(f"{color}: {percentage:.1f}% match")

def show_detection_debug(frame, territory, owner):
    x,y = territory.position
    # Draw territory marker
    cv2.circle(frame, (x,y), 10, (0,255,0), 2)
    # Show owner color
    color_map = {
        "red": (0,0,255),
        "blue": (255,0,0),
        # ... other colors
    }
    cv2.putText(frame, owner, (x+15,y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               color_map.get(owner, (255,255,255)), 2)
    
    # Show troop count
    troops = detect_troop_count(frame, territory)
    cv2.putText(frame, str(troops), (x,y-15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (0,255,255), 2)
    
    return frame

def detect_troop_count(frame, territory):
    x, y = territory.position
    # Increase the crop area slightly for better context
    crop = frame[y-25:y+25, x-25:x+25]
    
    if crop.size == 0:
        return 0
    
    # Convert to grayscale and apply preprocessing
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle varying lighting
    processed = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    # Configure Tesseract for better single digit recognition
    custom_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    count = pytesseract.image_to_string(processed, config=custom_config)
    
    # Additional validation
    count = count.strip()
    if not count.isdigit():
        # Try again with different preprocessing if first attempt fails
        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        count = pytesseract.image_to_string(processed, config=custom_config).strip()
    
    return int(count) if count.isdigit() else 0

def create_territories(frame) -> List[Territory]:
    """
    Creates 42 Territory instances with the collected pixel coordinates.
    Coordinates are assigned in the same order as territories are appended.
    """
    # List of collected coordinates in order
    coordinates = [
        (966, 715), (533, 321), (924, 441), (1116, 458), (883, 577),
        (699, 715), (1347, 738), (1312, 657), (748, 644), (577, 453),
        (1260, 511), (959, 655), (1150, 396), (971, 556), (890, 485),
        (835, 301), (1170, 546), (872, 424), (874, 365), (1404, 520),
        (744, 417), (664, 358), (1061, 679), (573, 525), (1234, 460),
        (1011, 433), (1406, 664), (653, 481), (1091, 418), (1053, 513),
        (677, 424), (685, 663), (963, 466), (1200, 362), (961, 382),
        (1006, 607), (1248, 670), (1251, 417), (594, 397), (684, 587),
        (1257, 569), (1344, 348)
    ]
    
    territories = []

    territories.append(Territory("Afríca do Sul", "África", 1, 1, "", coordinates[0]))
    territories.append(Territory("Alaska", "América do Norte", 2, 1, "", coordinates[1]))
    territories.append(Territory("Alemanha", "Europa", 3, 1, "", coordinates[2]))
    territories.append(Territory("Aral", "Asía", 4, 1, "", coordinates[3]))
    territories.append(Territory("Argélia", "África", 5, 1, "", coordinates[4]))

    territories.append(Territory("Argentina", "América do Sul", 6, 1, "", coordinates[5]))
    territories.append(Territory("Austrália", "Oceania", 7, 1, "", coordinates[6]))
    territories.append(Territory("Bornéu", "Oceania", 8, 1, "", coordinates[7]))
    territories.append(Territory("Brasil", "América do Sul", 9, 1, "", coordinates[8]))
    territories.append(Territory("Califórnia", "América do Norte", 10, 1, "", coordinates[9]))

    territories.append(Territory("China", "Asía", 11, 1, "", coordinates[10]))
    territories.append(Territory("Congo", "África", 12, 1, "", coordinates[11]))
    territories.append(Territory("Dudinka", "Asía", 13, 1, "", coordinates[12]))
    territories.append(Territory("Egito", "África", 14, 1, "", coordinates[13]))
    territories.append(Territory("França", "Europa", 15, 1, "", coordinates[14]))

    territories.append(Territory("Grôelandia", "América do Norte", 16, 1, "", coordinates[15]))
    territories.append(Territory("Índia", "Asía", 17, 1, "", coordinates[16]))
    territories.append(Territory("Inglaterra", "Europa", 18, 1, "", coordinates[17]))
    territories.append(Territory("Islândia", "Europa", 19, 1, "", coordinates[18]))
    territories.append(Territory("Japão", "Asía", 20, 1, "", coordinates[19]))

    territories.append(Territory("Labrador", "América do Norte", 21, 1, "", coordinates[20]))
    territories.append(Territory("Mackenzie", "América do Norte", 22, 1, "", coordinates[21]))
    territories.append(Territory("Madagascar", "África", 23, 1, "", coordinates[22]))
    territories.append(Territory("México", "América do Norte", 24, 1, "", coordinates[23]))
    territories.append(Territory("Mongólia", "Asía", 25, 1, "", coordinates[24]))

    territories.append(Territory("Moscou", "Europa", 26, 1, "", coordinates[25]))
    territories.append(Territory("Nova Guiné", "Oceania", 27, 1, "", coordinates[26]))
    territories.append(Territory("Nova York", "América do Norte", 28, 1, "", coordinates[27]))
    territories.append(Territory("Omsk", "Asía", 29, 1, "", coordinates[28]))
    territories.append(Territory("Oriente Médio", "Asía", 30, 1, "", coordinates[29]))

    territories.append(Territory("Ottawa", "América do Norte", 31, 1, "", coordinates[30]))
    territories.append(Territory("Peru", "América do Sul", 32, 1, "", coordinates[31]))
    territories.append(Territory("Polônia", "Europa", 33, 1, "", coordinates[32]))
    territories.append(Territory("Sibéria", "Asía", 34, 1, "", coordinates[33]))
    territories.append(Territory("Suécia", "Europa", 35, 1, "", coordinates[34]))

    territories.append(Territory("Sudão", "África", 36, 1, "", coordinates[35]))
    territories.append(Territory("Sumatra", "Oceania", 37, 1, "", coordinates[36]))    
    territories.append(Territory("Tchita", "Asía", 38, 1, "", coordinates[37]))
    territories.append(Territory("Vancouver", "América do Norte", 39, 1, "", coordinates[38]))
    territories.append(Territory("Venezuela", "América do Sul", 40, 1, "", coordinates[39]))

    territories.append(Territory("Vietnã", "Asía", 41, 1, "", coordinates[40]))
    territories.append(Territory("Vladvostok", "Asía", 42, 1, "", coordinates[41]))

    return territories

def scale_coordinates(coords, target_size):
    base_w, base_h = 1920, 1080  # Original reference size
    scale_x = target_size[0] / base_w
    scale_y = target_size[1] / base_h
    return [(int(x*scale_x), int(y*scale_y)) for x,y in coords]

# Usage:
# frame_size = (frame.shape[1], frame.shape[0])
# scaled_coords = scale_coordinates(coordinates, frame_size)

def create_cards() -> List[Card]:
    """
    Creates 42 Card instances explicitly, without loops, with names in alphabetical order.

    1 = Triangle
    2 = Square
    3 = Circle

    Returns:
        List[Card]: A list of 42 Card instances.
    """
    cards = []

    cards.append(Card("Afríca do Sul", 1, 0, 1))
    cards.append(Card("Alaska", 1, 0, 2))
    cards.append(Card("Alemanha", 3, 0, 3))
    cards.append(Card("Aral", 1, 0, 4))
    cards.append(Card("Argélia", 3, 0, 5))

    cards.append(Card("Argentina", 2, 0, 6))
    cards.append(Card("Austrália", 1, 0, 7))
    cards.append(Card("Bornéu", 2, 0, 8))
    cards.append(Card("Brasil", 3, 0, 9))
    cards.append(Card("Califórnia", 2, 0, 10))

    cards.append(Card("China", 3, 0, 11))
    cards.append(Card("Congo", 2, 0, 12))
    cards.append(Card("Dudinka", 3, 0, 13))
    cards.append(Card("Egito", 1, 0, 14))
    cards.append(Card("França", 2, 0, 15))

    cards.append(Card("Grôelandia", 3, 0, 16))
    cards.append(Card("Índia", 2, 0, 17))
    cards.append(Card("Inglaterra", 3, 0, 18))
    cards.append(Card("Islândia", 1, 0, 19))
    cards.append(Card("Japão", 2, 0, 20))

    cards.append(Card("Labrador", 2, 0, 21))
    cards.append(Card("Mackenzie", 3, 0, 22))
    cards.append(Card("Madagascar", 3, 0, 23))
    cards.append(Card("México", 2, 0, 25))
    cards.append(Card("Mongólia", 3, 0, 24))

    cards.append(Card("Moscou", 1, 0, 26))
    cards.append(Card("Nova Guiné", 3, 0, 27))
    cards.append(Card("Nova York", 1, 0, 28))
    cards.append(Card("Omsk", 2, 0, 29))
    cards.append(Card("Oriente Médio", 2, 0, 30))

    cards.append(Card("Ottawa", 3, 0, 31))
    cards.append(Card("Peru", 1, 0, 32))
    cards.append(Card("Polônia", 2, 0, 33))
    cards.append(Card("Sibéria", 1, 0, 34))
    cards.append(Card("Sudão", 2, 0, 36))    
    
    cards.append(Card("Suécia", 3, 0, 35))
    cards.append(Card("Sumatra", 2, 0, 37))
    cards.append(Card("Tchita", 1, 0, 38))
    cards.append(Card("Vancouver", 1, 0, 39))
    cards.append(Card("Venezuela", 1, 0, 40))

    cards.append(Card("Vietnã", 1, 0, 41))
    cards.append(Card("Vladvostok", 3, 0, 42))

    cards.append(Card("Coringa", 0, 0, 0))
    cards.append(Card("Coringa", 0, 0, 0))

    return cards

def assign_territories_to_players(players: List[Player], territories: List[Territory], num_players: int):
    """
    Assigns one territory to each player in a round-robin fashion.

    Args:
        players (List[Player]): List of Player instances.
        cards (List[Card]): List of 42 Card instances.
        territories (List[Territory]): List of 42 Territory instances.
        player_order (List[str]): The order in which players receive territories.
    """
    
    for i, territory in enumerate(territories):
        
        player = players[i % num_players]

        # Assign the territory to the player
        player.num_territories += 1
        player.num_armies += 1
        player.territories.append(territory.number)
        territory.player = player.player_color
        territory.armies += 1

def segment_by_color(frame, lower_hsv, upper_hsv):
    """
    Segments an image based on a specific color range in HSV.
    
    Parameters:
        frame (numpy.ndarray): Input image (BGR format).
        lower_hsv (tuple): Lower HSV threshold for color segmentation.
        upper_hsv (tuple): Upper HSV threshold for color segmentation.

    Returns:
        numpy.ndarray: Masked image with only the selected color.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply color threshold
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply mask to the original image
    color_segment = cv2.bitwise_and(frame, frame, mask=mask)

    return color_segment

def segment_region_by_edges(frame, x, y, image_res_scalar=1, canny_threshold1=100, canny_threshold2=200):
    """
    Segments a region based on edges, using an (x, y) point inside the region.

    Parameters:
        frame (numpy.ndarray): Input image (BGR format).
        x (int): X-coordinate of a point inside the target region.
        y (int): Y-coordinate of a point inside the target region.
        image_res_scalar (float): Scaling factor for coordinates (default = 1).
        canny_threshold1 (int): First threshold for Canny edge detection.
        canny_threshold2 (int): Second threshold for Canny edge detection.

    Returns:
        numpy.ndarray: Masked image with only the selected region.
    """
    # Scale coordinates
    x = int(x * image_res_scalar)
    y = int(y * image_res_scalar)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the selected region
    mask = np.zeros_like(gray)

    # Check which contour contains the (x, y) point
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            break  # Stop once we find the correct region

    # Apply the mask to extract only the selected region
    segmented_region = cv2.bitwise_and(frame, frame, mask=mask)

    return segmented_region

def calibrate_rgb_detection(frame, territory, white_threshold=230):
    """
    Interactive tool to find optimal RGB ranges while ignoring white pixels
    Args:
        frame: Input image (BGR format)
        territory: Territory object with position attribute
        white_threshold: BGR values above this will be considered white (0-255)
    Returns:
        Tuple of (lower_bound, upper_bound) in BGR format
    """
    x, y = territory.position
    sample_radius = 10
    region = frame[y-sample_radius:y+sample_radius, 
                  x-sample_radius:x+sample_radius]
    
    # Create mask for non-white pixels
    mask = np.all(region < white_threshold, axis=2)
    
    print(f"\nCalibrating for {territory.name} at ({x},{y})")
    print(f"Sample region size: {region.shape[0]}x{region.shape[1]}")
    print(f"Non-white pixels: {np.count_nonzero(mask)}/{mask.size}")

    if np.count_nonzero(mask) == 0:
        print("Warning: Only white pixels found in sample region!")
        return (0, 0, 0), (255, 255, 255)

    # Calculate color statistics only on non-white pixels
    non_white_pixels = region[mask]
    avg_color = np.mean(non_white_pixels, axis=0)
    std_color = np.std(non_white_pixels, axis=0)
    
    print("\nNon-white pixels RGB stats:")
    print(f"Average BGR: {avg_color}")
    print(f"Std Dev: {std_color}")
    
    # Suggested initial bounds (±2 standard deviations)
    lower = np.clip(avg_color - 2*std_color, 0, 255).astype(int)
    upper = np.clip(avg_color + 2*std_color, 0, 255).astype(int)
    
    print("\nSuggested RGB range (white ignored):")
    print(f"Lower: {tuple(lower)}")
    print(f"Upper: {tuple(upper)}")
    
    # Enhanced visual feedback
    debug_img = region.copy()
    debug_img[~mask] = 0  # Black out white pixels
    
    cv2.imshow(f"Sample Region: {territory.name}", debug_img)
    cv2.waitKey(2000)  # Show for 2 seconds
    
    return tuple(lower), tuple(upper)

def find_reference_points(frame):
    """Detect stable reference points on the game map"""
    # Example: Detect corners of the game board or UI elements
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    return corners.reshape(-1, 2) if corners is not None else None

def calculate_transform(base_frame, current_frame):
    """Calculate homography transform between frames"""
    base_refs = find_reference_points(base_frame)
    current_refs = find_reference_points(current_frame)
    
    if base_refs is None or current_refs is None or len(base_refs) != 4:
        return None  # Fallback to default positions
    
    # Order points consistently (top-left, top-right, bottom-right, bottom-left)
    base_refs = order_points(base_refs)
    current_refs = order_points(current_refs)
    
    # Calculate homography matrix
    H, _ = cv2.findHomography(base_refs, current_refs)
    return H

def order_points(pts):
    """Arrange points in consistent order"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def adjust_territory_positions(territories, H, frame_shape):
    """Apply perspective transform to territory positions"""
    adjusted = []
    for territory in territories:
        # Convert to homogeneous coordinates
        x, y = territory.position
        pt = np.array([[x, y]], dtype='float32')
        pt = np.array([pt])
        
        # Apply perspective transform
        transformed = cv2.perspectiveTransform(pt, H)
        new_x, new_y = transformed[0][0]
        
        # Ensure positions stay within frame bounds
        new_x = np.clip(new_x, 0, frame_shape[1]-1)
        new_y = np.clip(new_y, 0, frame_shape[0]-1)
        
        # Create new territory with adjusted position
        adjusted.append(Territory(
            name=territory.name,
            continent=territory.continent,
            number=territory.number,
            armies=territory.armies,
            player=territory.player,
            position=(int(new_x), int(new_y))
        ))
    return adjusted

def detect_view_changes(H):
    """Analyze homography matrix for zoom/pan"""
    if H is None:
        return "unknown", 1.0
    
    # Extract scale (approximate)
    scale_x = np.sqrt(H[0,0]**2 + H[0,1]**2)
    scale_y = np.sqrt(H[1,0]**2 + H[1,1]**2)
    avg_scale = (scale_x + scale_y) / 2
    
    # Detect pan
    pan_x = H[0,2]
    pan_y = H[1,2]
    
    return (pan_x, pan_y), avg_scale

def get_color_samples(image_path, coordinates):
    img = cv2.imread(image_path)
    samples = {}
    for name, (x,y) in coordinates.items():
        # Extract 5x5 pixel sample
        sample = img[y-2:y+3, x-2:x+3]  # 5x5 sample around coordinate
        samples[name] = sample
    return samples

def analyze_samples(samples):
    color_ranges = {}
    for name, sample in samples.items():
        # Reshape to pixels x RGB
        pixels = sample.reshape(-1, 3)
        
        # Filter out white/light pixels (BGR values all > threshold)
        white_threshold = 230  # Adjust this value as needed
        is_not_white = np.any(pixels < white_threshold, axis=1)
        filtered = pixels[is_not_white]
        
        if len(filtered) > 0:
            avg = np.mean(filtered, axis=0)
            std = np.std(filtered, axis=0)
            lower = np.clip(avg - 2*std, 0, 255).astype(int)
            upper = np.clip(avg + 2*std, 0, 255).astype(int)
            color_ranges[name] = (tuple(lower), tuple(upper))
    return color_ranges

def mouse_callback_game_state(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, territories, players = param  # Unpack the parameters we passed
        game_state = extract_game_state(frame, territories, players)
        print_game_state(game_state, players)

def mouse_callback_rgb_calibration(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, territories, players = param  # Unpack the parameters we passed
        
    # Dictionary to store calibrated ranges
        color_ranges = {}
        
        # Calibrate each territory
        for territory in territories:
            print(f"\nCalibrating {territory.name}...")
            ranges = calibrate_rgb_detection(frame, territory)
            
            if ranges:
                color_ranges[territory.name] = ranges
                print(f"Saved range: {ranges}")
            else:
                print(f"Skipped {territory.name} due to error")
        
        # Show final calibration results
        print("\n=== Final Calibration Results ===")
        for name, (lower, upper) in color_ranges.items():
            print(f"{name}:")
            print(f"  BGR Range: {lower} to {upper}")
            print(f"  Preview:")

# During initialization:
base_frame = cv2.imread("war_reference_ss_1.png")
base_territories = create_territories(base_frame)

# Load the video file
# video_path = "war_match_1.mp4"   # FRAME QUE COMEÇA A PARTIDA 1 DE FATO: 1000
# cap = cv2.VideoCapture(video_path)

# # Check if the video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

win_name = "War Game"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

frame_number = 0  # Initialize frame counter

trade_number = 0

num_players = 6

# Step 1: Initialize players
players = initialize_players(num_players)

# Step 2: Create 42 cards with territory names
cards = create_cards()

# ret_dummy, frame_scale = cap.read()

# Step 4: Create 42 territories with territory names
territories = create_territories(base_frame)

# # Step 5: Assign territories to players
# assign_territories_to_players(players, territories, num_players)

# # Step 6: Assign objectives
# get_objectives(players)


# VIDEO PROCESSING

# while True:
#     ret, frame = cap.read()
#     if not ret:  # Break if video ends
#         break

#     # cv2.setMouseCallback("War Game", mouse_callback, (frame.copy(), territories, players))
#     cv2.setMouseCallback("War Game", mouse_callback_rgb_calibration, (frame.copy(), territories, players))

#     frame_number += 1  # Increment frame counter

#     # Add frame number text overlay
#     text = f"Frame: {frame_number}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     position = (50, 50)  # Position (x, y)
#     font_scale = 1
#     font_color = (0, 255, 0)  # Green color
#     thickness = 2
#     line_type = cv2.LINE_AA

#     cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)

#     # Calculate transformation
#     H = calculate_transform(base_frame, frame)
    
#     if H is not None:
#         # Adjust territory positions dynamically
#         current_territories = adjust_territory_positions(
#             base_territories, 
#             H, 
#             frame.shape
#         )
#     else:
#         # Fallback to original positions
#         current_territories = base_territories

#     # Show the frame
#     cv2.imshow(win_name, frame)

# IMAGE PROCESSING

# Load your single image
image_path = "war_reference_ss_1.png"  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

cv2.setMouseCallback("War Game", mouse_callback_game_state, (frame.copy(), territories, players))

# Show the frame

cv2.imshow(win_name, frame)

# coordinates = {
#     "Afríca do Sul": (966, 715),
#     "Alaska": (533, 321),
#     "Alemanha": (924, 441),
#     "Aral": (1116, 458),
#     "Argélia": (883, 577),
#     "Argentina": (699, 715),
#     "Austrália": (1347, 738),
#     "Bornéu": (1312, 657),
#     "Brasil": (748, 644),
#     "Califórnia": (577, 453),
#     "China": (1260, 511),
#     "Congo": (959, 655),
#     "Dudinka": (1150, 396),
#     "Egito": (971, 556),
#     "França": (890, 485),
#     "Grôelandia": (835, 301),
#     "Índia": (1170, 546),
#     "Inglaterra": (872, 424),
#     "Islândia": (874, 365),
#     "Japão": (1404, 520),
#     "Labrador": (744, 417),
#     "Mackenzie": (664, 358),
#     "Madagascar": (1061, 679),
#     "México": (573, 525),
#     "Mongólia": (1234, 460),
#     "Moscou": (1011, 433),
#     "Nova Guiné": (1406, 664),
#     "Nova York": (653, 481),
#     "Omsk": (1091, 418),
#     "Oriente Médio": (1053, 513),
#     "Ottawa": (677, 424),
#     "Peru": (685, 663),
#     "Polônia": (963, 466),
#     "Sibéria": (1200, 362),
#     "Suécia": (961, 382),
#     "Sudão": (1006, 607),
#     "Sumatra": (1248, 670),
#     "Tchita": (1251, 417),
#     "Vancouver": (594, 397),
#     "Venezuela": (684, 587),
#     "Vietnã": (1257, 569),
#     "Vladvostok": (1344, 348)
# }

# samples = get_color_samples("war_reference_ss_1.png", coordinates)
# color_ranges = analyze_samples(samples)

# # Print results
# for territory, (lower, upper) in color_ranges.items():
#     print(f"{territory}:")
#     print(f"  Lower: {lower}")
#     print(f"  Upper: {upper}")
#     print("-" * 40)

    # frame_edges = segment_region_by_edges(frame, 279, 175)

    # cv2.imshow("Edges", frame_edges)

    # frame_color_green = segment_by_color(frame, green_lower_hsv, green_upper_hsv)

    # cv2.imshow("Green", frame_color_green)

    # frame_color_yellow = segment_by_color(frame, yellow_lower_hsv, yellow_upper_hsv)

    # cv2.imshow("Yellow", frame_color_yellow)

    # frame_color_blue = segment_by_color(frame, blue_lower_hsv, blue_upper_hsv)

    # cv2.imshow("Blue", frame_color_blue)

    # frame_color_purple = segment_by_color(frame, purple_lower_hsv, purple_upper_hsv)

    # cv2.imshow("Purple", frame_color_purple)

    # frame_color_gray = segment_by_color(frame, gray_lower_hsv, gray_upper_hsv)

    # cv2.imshow("Gray", frame_color_gray)

    # frame_color_red_1 = segment_by_color(frame, red1_lower_hsv, red1_upper_hsv)
    # frame_color_red_2 = segment_by_color(frame, red2_lower_hsv, red2_upper_hsv)

    # frame_color_red_both = cv2.bitwise_or(frame_color_red_1, frame_color_red_2)

    # cv2.imshow("Red", frame_color_red_both)

    # Press 'q' to exit
    # Wait for key press to exit

while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release resources
# cap.release()
cv2.destroyAllWindows()
