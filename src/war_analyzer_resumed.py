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

def mouse_callback_game_state(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, territories, players = param  # Unpack the parameters we passed
        game_state = extract_game_state(frame, territories, players)
        print_game_state(game_state, players)

# During initialization:
# base_frame = cv2.imread("war_reference_ss_1.png")
# base_territories = create_territories(base_frame)

win_name = "War Game"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# IMAGE PROCESSING

# Load your single image
image_path = "war_reference_ss_1.png"  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Initialize frame counter
frame_number = 0 

# Initialize in which trade the game is
trade_number = 0

# Define the number of players
num_players = 6

# Step 1: Initialize players
players = initialize_players(num_players)

# Step 2: Create 42 cards with territory names
cards = create_cards()

# Step 4: Create 42 territories with territory names
territories = create_territories(frame)

# # Step 5: Assign territories to players
# assign_territories_to_players(players, territories, num_players)

# # Step 6: Assign objectives
# get_objectives(players)

cv2.setMouseCallback("War Game", mouse_callback_game_state, (frame.copy(), territories, players))

# Show the frame

cv2.imshow(win_name, frame)

while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
