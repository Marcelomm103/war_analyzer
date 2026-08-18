import random
import cv2
from typing import Tuple, List
import numpy as np
import pytesseract
import mss # For screen capture
from pynput import keyboard # For global key listener
import time # For managing capture loop

REFERENCE_ANCHOR_POS_X, REFERENCE_ANCHOR_POS_Y = 1423, 387 # Example: where your anchor was when you set original_position

class Territory:
    def __init__(self, name: str, continent: str, number: int, armies: int, player: str, original_position: Tuple[int, int], anchor_offset: Tuple[int, int] = None): # original_position is screen absolute at reference
        self.name = name
        self.continent = continent
        self.number = number
        self.armies = armies
        self.player = player
        self.original_position = original_position # The coords when map is at a reference position
        self.current_position = original_position # Will be updated each frame
        # self.anchor_offset = anchor_offset # (tx - anchor_x, ty - anchor_y) IF you pre-calculate this


class Player:
    def __init__(self, name: str, player_color: str):
        self.name = name # You might want to set this to player_color initially
        self.player_color = player_color
        self.num_armies = 0
        self.num_territories = 0
        self.territories = [] # List of territory numbers or names
        self.objective = None
        self.bonus_armies = 0
        self.cards: List['Card'] = []


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
    if num_players < 3 or num_players > 6:
        raise ValueError("Number of players must be between 3 and 6.")

    available_colors = ["blue", "red", "yellow", "green", "gray", "purple"]
    # Ensure the game's actual color names match these keys if used for indexing
    # For the provided screenshot, the colors are:
    # green, purple, red, yellow, a cyan-like blue, and a dark gray
    # The keys in rgb_ranges should match player.player_color
    
    # Let's ensure we have enough unique colors from the available_colors list
    if num_players > len(available_colors):
        raise ValueError(f"Not enough unique colors defined for {num_players} players.")
        
    selected_colors = random.sample(available_colors, num_players)
    players = [Player(name=color, player_color=color) for color in selected_colors] # Set name to color
    return players

# ... (keep initialize_objectives, get_objectives)
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
    
    # Filter objectives related to non-participating colors if num_players < 6
    # This part of the rules is important: "no caso do número de jogadores ser inferior a 6, os objetivos relacionados com os exércitos não participantes devem ser excluídos do sorteio"
    if len(players) < 6:
        active_player_colors = {player.player_color for player in players}
        filtered_objectives = []
        # Assuming objectives 9-14 are "Destroy color X"
        color_to_objective_idx_map = {
            "green": 9, "yellow": 10, "purple": 11, "gray": 12, "red": 13, "blue": 14
        }
        destroy_objective_indices = set(color_to_objective_idx_map.values())

        for obj in objectives:
            is_destroy_objective = obj.obj_number in destroy_objective_indices
            if is_destroy_objective:
                target_color = None
                for color, idx in color_to_objective_idx_map.items():
                    if idx == obj.obj_number:
                        target_color = color
                        break
                if target_color in active_player_colors:
                    filtered_objectives.append(obj)
            else: # Not a "destroy specific color" objective
                filtered_objectives.append(obj)
        objectives = filtered_objectives
        
    if len(objectives) < len(players):
        # This can happen if too many destroy objectives are filtered out.
        # The game rules might need a fallback here, or ensure enough generic objectives.
        # For now, we'll proceed, but it might cause an error if objectives run out.
        print(f"Warning: Not enough objectives ({len(objectives)}) for players ({len(players)}) after filtering.")

    random.shuffle(objectives)

    for player in players:
        if not objectives:
            print(f"Warning: Ran out of objectives to assign to player {player.name}")
            # Assign a default or handle this scenario based on game rules for this edge case
            player.objective = Objective(0, "Default: Conquer 24 territories (ran out of unique objectives)")
            continue
        player.objective = objectives.pop()

def detect_owner_rgb(frame, territory, players):
    """Detect owner using RGB color ranges while ignoring white pixels"""
    x, y = territory.current_position
    sample_radius = 10
    min_match_percent = 0.3
    white_threshold = 230 # BGR threshold for white

    y1, y2 = max(0, y - sample_radius), min(frame.shape[0], y + sample_radius)
    x1, x2 = max(0, x - sample_radius), min(frame.shape[1], x + sample_radius)
    region = frame[y1:y2, x1:x2]
    
    if region.size == 0:
        return "neutral"
    
    white_mask_inv = cv2.inRange(region, 
                                (white_threshold, white_threshold, white_threshold),
                                (255, 255, 255))
    non_white_pixels_mask = cv2.bitwise_not(white_mask_inv)
    total_non_white = cv2.countNonZero(non_white_pixels_mask)
    
    if total_non_white == 0:
        return "neutral"
    
    # BGR format. These ranges seem well-calibrated to your screenshot.
    # "blue" here corresponds to the cyan-like color in the game.
    rgb_ranges = {
        "red": ((22, 0, 186), (32, 100, 196)),      # Actual Red
        "blue": ((178, 159, 0), (188, 169, 0)),     # Cyan / Teal
        "green": ((45, 165, 26), (55, 175, 36)),    # Green
        "yellow": ((0, 190, 213), (0, 200, 223)),   # Yellow
        "purple": ((110, 0, 82), (120, 0, 92)),   # Purple / Magenta
        "gray": ((51, 51, 51), (61, 61, 61))      # Dark Gray
    }
    
    match_counts = {}
    for player in players:
        if player.player_color not in rgb_ranges: # Should not happen if colors are from the list
            continue 
        lower = np.array(rgb_ranges[player.player_color][0], dtype="uint8")
        upper = np.array(rgb_ranges[player.player_color][1], dtype="uint8")
        
        color_mask = cv2.inRange(region, lower, upper)
        final_mask = cv2.bitwise_and(color_mask, non_white_pixels_mask)
        match_counts[player.player_color] = cv2.countNonZero(final_mask)
    
    if not match_counts: # No player colors matched or no players provided
        return "neutral"

    best_match_color, best_match_count = max(match_counts.items(), key=lambda item: item[1])
    
    if (best_match_count / total_non_white) >= min_match_percent:
        return best_match_color
    return "neutral"


def detect_troop_count(frame, territory, debug=False, territory_name_for_debug=""):

    x, y = territory.current_position    
    
    # 1. Precise Cropping
    # These offsets define the size of the box around the number's center (x,y)
    # y_offset: half-height of the box, x_offset: half-width of the box
    y_offset = 16  # Crop height will be 2 * 12 = 24 pixels
    x_offset = 24  # Crop width will be 2 * 18 = 36 pixels (to accommodate two digits)

    y1, y2 = max(0, y - y_offset), min(frame.shape[0], y + y_offset)
    x1, x2 = max(0, x - x_offset), min(frame.shape[1], x + x_offset)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        if debug: print(f"Debug OCR - {territory_name_for_debug}: Crop is empty.")
        return 0

    if debug:
        cv2.imwrite(f"debug_images/debug_ocr_01_crop_{territory_name_for_debug}.png", crop)

    # 2. Scaling for better OCR
    scale_factor = 3.0 # Upscale by 3x. Numbers become ~30-40px high.
    width = int(crop.shape[1] * scale_factor)
    height = int(crop.shape[0] * scale_factor)
    
    if width <= 0 or height <= 0:
        if debug: print(f"Debug OCR - {territory_name_for_debug}: Invalid crop dimensions for scaling: w={width}, h={height}")
        return 0
        
    resized_crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

    if debug:
        cv2.imwrite(f"debug_images/debug_ocr_02_resized_{territory_name_for_debug}.png", resized_crop)

    # 3. White Color Masking (NEW APPROACH)
    # The numbers are bright white. Define a BGR range for this.
    # If numbers on yellow appear slightly tinted, this range might need adjustment,
    # particularly the lower bound for the blue channel if there's a yellow cast.
    # Start with a fairly strict "bright white" range.
    lower_white = np.array([215, 215, 215], dtype=np.uint8) # Pixels must be quite bright in all channels
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    
    # Create a mask. Pixels within the white range become 255 (white), others 0 (black).
    # This `initial_white_mask` should have white numbers on a black background.
    initial_white_mask = cv2.inRange(resized_crop, lower_white, upper_white)
    
    if debug:
        cv2.imwrite(f"debug_images/debug_ocr_03a_initial_white_mask_{territory_name_for_debug}.png", initial_white_mask)

    # 4. Morphological Operations to clean the mask
    # Kernel size: On the 3x scaled image, features are larger.
    # A (2,2) or (3,3) kernel is usually good for minor cleanup.
    # Let's try a (3,3) kernel.
    kernel_size = (3,3) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Perform operations on `initial_white_mask` (white numbers, black background):
    # - Closing: Fill small black holes inside the white number shapes.
    # - Opening: Remove small white noise speckles around/detached from numbers.

    # Step 1: Closing to make digits more solid
    cleaned_mask = cv2.morphologyEx(initial_white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if debug:
        cv2.imwrite(f"debug_images/debug_ocr_03b_closed_mask_{territory_name_for_debug}.png", cleaned_mask)
        
    # Step 2: Opening to remove any remaining small noise islands
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if debug:
        cv2.imwrite(f"debug_images/debug_ocr_03c_opened_mask_{territory_name_for_debug}.png", cleaned_mask)

    # `cleaned_mask` now has (hopefully) clean white numbers on a black background.
    # Tesseract generally prefers black text on a white background. So, invert the mask.
    processed_for_ocr = cv2.bitwise_not(cleaned_mask)

    # --- Experiment Start ---
    # Option A: Make text slightly thicker
    # kernel_dilate = np.ones((2,2), np.uint8) # Small kernel
    # processed_for_ocr = cv2.dilate(processed_for_ocr, kernel_dilate, iterations=1)

    # Option B: Make text slightly thinner
    kernel_erode = np.ones((6,6), np.uint8)
    processed_for_ocr = cv2.erode(processed_for_ocr, kernel_erode, iterations=1)

    # Option D: Add padding
    # border_size = 8 # pixels
    # processed_for_ocr_final = cv2.copyMakeBorder(
    #     processed_for_ocr,
    #     top=border_size, bottom=border_size, left=border_size, right=border_size,
    #     borderType=cv2.BORDER_CONSTANT,
    #     value=[255,255,255] # White padding
    # )
    # --- Experiment End ---
    
    if debug:
        cv2.imwrite(f"debug_images/debug_ocr_04_final_processed_for_tesseract_{territory_name_for_debug}.png", processed_for_ocr)

    # 5. Tesseract Configuration & OCR
    # --psm 6: Assume a single uniform block of text. (Good for isolated numbers)
    # --oem 1: Use LSTM OCR engine only.
    custom_config = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789'
    
    text = pytesseract.image_to_string(processed_for_ocr, config=custom_config)
    count_str = ''.join(filter(str.isdigit, text)) # Keep only digits

    if debug:
        print(f"Debug OCR - {territory_name_for_debug}: Raw Tesseract (psm 6): '{text.strip()}', Cleaned Digits: '{count_str}'")

    # Fallback strategy if psm 6 fails or returns non-digits
    if not count_str.isdigit():
        # Try --psm 7: Treat the image as a single text line.
        custom_config_fallback_psm7 = r'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789'
        text_fallback_psm7 = pytesseract.image_to_string(processed_for_ocr, config=custom_config_fallback_psm7)
        count_str_psm7 = ''.join(filter(str.isdigit, text_fallback_psm7))
        if debug:
            print(f"Debug OCR - {territory_name_for_debug} (Fallback psm 7): Raw: '{text_fallback_psm7.strip()}', Cleaned: '{count_str_psm7}'")
        if count_str_psm7.isdigit():
            return int(count_str_psm7)
        else:
            # Final fallback: Try --psm 10 (single character) - useful if '1's are common and other modes fail
            custom_config_fallback_psm10 = r'--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789'
            text_fallback_psm10 = pytesseract.image_to_string(processed_for_ocr, config=custom_config_fallback_psm10)
            count_str_psm10 = ''.join(filter(str.isdigit, text_fallback_psm10))
            if debug:
                print(f"Debug OCR - {territory_name_for_debug} (Fallback psm 10): Raw: '{text_fallback_psm10.strip()}', Cleaned: '{count_str_psm10}'")
            if count_str_psm10.isdigit():
                return int(count_str_psm10)
            return 0 # Default to 0 if all attempts fail
    
    return int(count_str)

def extract_game_state(frame, territories_list, players, debug_ocr=False): # Takes territories_list
    game_state = {}
    print("\nExtracting Game State...")
    for i, territory_obj in enumerate(territories_list): # Iterate through the list of Territory objects
        # ... (clean name)
        # Pass the territory_obj itself
        troops = detect_troop_count(frame, territory_obj, debug=debug_ocr, territory_name_for_debug=territory_obj.name)
        owner_color_str = detect_owner_rgb(frame, territory_obj, players)

        territory_obj.armies = troops # Update the object
        territory_obj.player = owner_color_str

        print(f"Territory: {territory_obj.name:<20} (at {territory_obj.current_position}) | Troops: {troops:<3} | Owner: {owner_color_str}")
        
        game_state[territory_obj.name] = {
            "owner": owner_color_str,
            "troops": troops,
            "position": territory_obj.current_position, # Store the adjusted position
            "continent": territory_obj.continent
        }
    # ...
    return game_state

def print_game_state(game_state, players):
    player_colors = [player.player_color for player in players]
    player_territories = {color: [] for color in player_colors}
    player_territories["neutral"] = [] 
    
    for territory_name, data in game_state.items():
        owner = data["owner"].lower() 
        if owner not in player_territories: # Should only be neutral if not in player_colors
            owner = "neutral"
        player_territories[owner].append(
            (territory_name, data["troops"], data["continent"])
        )
    
    for player in players:
        color = player.player_color
        territories_list = player_territories.get(color, []) # Use .get for safety
        
        print(f"\nPlayer {player.name} ({color}) - Territories: {len(territories_list)}")
        print("-" * 40)
        
        continents = {}
        for terr_name, troops, continent_name in territories_list:
            if continent_name not in continents:
                continents[continent_name] = []
            continents[continent_name].append(f"{terr_name} ({troops})")
        
        for continent_name, terr_list_str in continents.items():
            print(f"  {continent_name}:")
            print("    " + ", ".join(terr_list_str))
            
    neutral_terr = player_territories["neutral"]
    if neutral_terr:
        print("\nNeutral/Unknown Territories:")
        print("-" * 40)
        continents_neutral = {}
        for terr_name, troops, continent_name in neutral_terr:
            if continent_name not in continents_neutral:
                continents_neutral[continent_name] = []
            continents_neutral[continent_name].append(f"{terr_name} ({troops})")
        for continent_name, terr_list_str in continents_neutral.items():
            print(f"  {continent_name}:")
            print("    " + ", ".join(terr_list_str))

def create_territories(frame_shape_for_reference) -> List[Territory]: # Pass frame shape if you plan to scale

    original_coordinates_data = [
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
    
    territories_data = [
        ("Afríca do Sul", "África", 1), ("Alaska", "América do Norte", 2), ("Alemanha", "Europa", 3),
        ("Aral", "Asía", 4), ("Argélia", "África", 5), ("Argentina", "América do Sul", 6),
        ("Austrália", "Oceania", 7), ("Bornéu", "Oceania", 8), ("Brasil", "América do Sul", 9),
        ("Califórnia", "América do Norte", 10), ("China", "Asía", 11), ("Congo", "África", 12),
        ("Dudinka", "Asía", 13), ("Egito", "África", 14), ("França", "Europa", 15),
        ("Grôelandia", "América do Norte", 16), ("Índia", "Asía", 17), ("Inglaterra", "Europa", 18),
        ("Islândia", "Europa", 19), ("Japão", "Asía", 20), ("Labrador", "América do Norte", 21),
        ("Mackenzie", "América do Norte", 22), ("Madagascar", "África", 23), ("México", "América do Norte", 24),
        ("Mongólia", "Asía", 25),("Moscou", "Europa", 26), ("Nova Guiné", "Oceania", 27),
        ("Nova York", "América do Norte", 28), ("Omsk", "Asía", 29), ("Oriente Médio", "Asía", 30),
        ("Ottawa", "América do Norte", 31), ("Peru", "América do Sul", 32), ("Polônia", "Europa", 33),
        ("Sibéria", "Asía", 34), ("Suécia", "Europa", 35), ("Sudão", "África", 36),
        ("Sumatra", "Oceania", 37), ("Tchita", "Asía", 38), ("Vancouver", "América do Norte", 39),
        ("Venezuela", "América do Sul", 40), ("Vietnã", "Asía", 41), ("Vladvostok", "Asía", 42)
    ]

    territories_list = []
    for i, (name, continent, number) in enumerate(territories_data):
        orig_pos = original_coordinates_data[i]
        offset_x = orig_pos[0] - REFERENCE_ANCHOR_POS_X
        offset_y = orig_pos[1] - REFERENCE_ANCHOR_POS_Y
        terr = Territory(name, continent, number, 0, "", orig_pos, anchor_offset=(offset_x, offset_y))
        territories_list.append(terr)
    return territories_list

def create_cards() -> List[Card]:
    # ... (your create_cards function is fine) ...
    cards = []
    cards.append(Card("Afríca do Sul", 1, 0, 1)); cards.append(Card("Alaska", 1, 0, 2))
    cards.append(Card("Alemanha", 3, 0, 3)); cards.append(Card("Aral", 1, 0, 4))
    cards.append(Card("Argélia", 3, 0, 5)); cards.append(Card("Argentina", 2, 0, 6))
    cards.append(Card("Austrália", 1, 0, 7)); cards.append(Card("Bornéu", 2, 0, 8))
    cards.append(Card("Brasil", 3, 0, 9)); cards.append(Card("Califórnia", 2, 0, 10))
    cards.append(Card("China", 3, 0, 11)); cards.append(Card("Congo", 2, 0, 12))
    cards.append(Card("Dudinka", 3, 0, 13)); cards.append(Card("Egito", 1, 0, 14))
    cards.append(Card("França", 2, 0, 15)); cards.append(Card("Grôelandia", 3, 0, 16))
    cards.append(Card("Índia", 2, 0, 17)); cards.append(Card("Inglaterra", 3, 0, 18))
    cards.append(Card("Islândia", 1, 0, 19)); cards.append(Card("Japão", 2, 0, 20))
    cards.append(Card("Labrador", 2, 0, 21)); cards.append(Card("Mackenzie", 3, 0, 22))
    cards.append(Card("Madagascar", 3, 0, 23)); cards.append(Card("México", 2, 0, 25)) # Original had 25, Mongólia 24. Check numbering. Assuming México is 24, Mongólia 25 from typical lists.
    cards.append(Card("Mongólia", 3, 0, 24)); # Corrected territory number if México is 24. Let's use your original if that's intended.
    # Reverting to your original numbering for Mexico and Mongolia cards to match your territory list:
    # cards.append(Card("México", 2, 0, 24)) # As per your Territory list
    # cards.append(Card("Mongólia", 3, 0, 25)) # As per your Territory list
    # For consistency, I'll use the provided territory numbers from create_territories for cards:
    # México is 24, Mongólia is 25.
    cards.append(Card("México", 2, 0, 24))
    cards.append(Card("Mongólia", 3, 0, 25))

    cards.append(Card("Moscou", 1, 0, 26)); cards.append(Card("Nova Guiné", 3, 0, 27))
    cards.append(Card("Nova York", 1, 0, 28)); cards.append(Card("Omsk", 2, 0, 29))
    cards.append(Card("Oriente Médio", 2, 0, 30)); cards.append(Card("Ottawa", 3, 0, 31))
    cards.append(Card("Peru", 1, 0, 32)); cards.append(Card("Polônia", 2, 0, 33))
    cards.append(Card("Sibéria", 1, 0, 34)); cards.append(Card("Suécia", 3, 0, 35)) # Sudao was 36, Suecia 35
    cards.append(Card("Sudão", 2, 0, 36)); 
    cards.append(Card("Sumatra", 2, 0, 37)); cards.append(Card("Tchita", 1, 0, 38))
    cards.append(Card("Vancouver", 1, 0, 39)); cards.append(Card("Venezuela", 1, 0, 40))
    cards.append(Card("Vietnã", 1, 0, 41)); cards.append(Card("Vladvostok", 3, 0, 42))
    cards.append(Card("Coringa", 0, 0, 0)); cards.append(Card("Coringa", 0, 0, 0)) # Jokers
    return cards

# --- (Keep assign_territories_to_players if needed for initializing a simulated game) ---
# --- (The scale_coordinates function is fine, but ensure base_w, base_h match your reference screenshot if you use it) ---
# Base screenshot war_reference_ss_1.png is 1600x900.
# def scale_coordinates(coords, target_size):
#     base_w, base_h = 1600, 900  # Original reference size of war_reference_ss_1.png
#     scale_x = target_size[0] / base_w
#     scale_y = target_size[1] / base_h
#     return [(int(x*scale_x), int(y*scale_y)) for x,y in coords]

def find_anchor_and_adjust_territories(frame, territories_list, anchor_template, 
                                       reference_anchor_position_if_known=None):
    """
    Finds the anchor template in the frame and adjusts territory positions.
    Returns True if anchor found and positions adjusted, False otherwise.
    Modifies territory.current_position in place.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # If your anchor_template is grayscale, use it directly
    # template_gray = cv2.cvtColor(anchor_template, cv2.COLOR_BGR2GRAY) # if anchor_template is BGR
    template_gray = anchor_template # If ANCHOR_TEMPLATE was loaded as grayscale

    res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    threshold = 0.8 # Adjust this threshold based on how well your template matches
    if max_val >= threshold:
        # max_loc is the top-left of the found anchor
        current_anchor_x, current_anchor_y = max_loc

        # For debugging: draw a rectangle around the found anchor
        # anchor_w, anchor_h = template_gray.shape[::-1] # width, height
        # cv2.rectangle(frame, (current_anchor_x, current_anchor_y), 
        #               (current_anchor_x + anchor_w, current_anchor_y + anchor_h), (0,255,0), 2)
        # print(f"Anchor found at: ({current_anchor_x}, {current_anchor_y}) with confidence {max_val:.2f}")


        # --- Method 1: Using a known reference anchor position ---
        if reference_anchor_position_if_known:
            ref_ax, ref_ay = reference_anchor_position_if_known
            displacement_x = current_anchor_x - ref_ax
            displacement_y = current_anchor_y - ref_ay

            for terr in territories_list:
                terr.current_position = (
                    terr.original_position[0] + displacement_x,
                    terr.original_position[1] + displacement_y
                )
        # --- Method 2: Using pre-calculated offsets (if terr.anchor_offset was stored) ---
        # else:
        #     for terr in territories_list:
        #         if terr.anchor_offset:
        #             # current_anchor_x/y is top-left. If offsets are from anchor center, adjust.
        #             # Assuming offsets are from anchor's top-left for simplicity here.
        #             terr.current_position = (
        #                 current_anchor_x + terr.anchor_offset[0],
        #                 current_anchor_y + terr.anchor_offset[1]
        #             )
        #         else: # Fallback if no offset, maybe use original (less accurate if map moved)
        #             terr.current_position = terr.original_position
        else:
            # This else implies you don't have a reference anchor position.
            # This method is more complex because your original_position must have been
            # defined relative to an anchor at (0,0) of the map itself, or you
            # need to establish the offsets during a setup phase.
            # For now, let's assume Method 1 or that original_positions are good enough
            # if no reference_anchor_position_if_known. This needs careful thought
            # about your coordinate system's origin.
            # A simple approach if no reference_anchor_position:
            # Assume the first territory's original_position was relative to an anchor
            # that was at some (dx, dy) from it. This is tricky without a clear setup.
            #
            # **Let's simplify: For now, assume we MUST provide a reference_anchor_position_if_known**
            # Or, if not provided, the original coordinates are used (map hasn't moved from reference)
            if not reference_anchor_position_if_known:
                 print("Warning: No reference anchor position provided. Using original territory coordinates.")
                 for terr in territories_list:
                     terr.current_position = terr.original_position

        return True
    else:
        print(f"Anchor not found or confidence too low ({max_val:.2f} < {threshold}). Using last known/original positions.")
        # Optionally, revert to original positions or last known good positions
        for terr in territories_list:
            terr.current_position = terr.original_position # Or keep last frame's current_position
        return False


def mouse_callback_game_state(event, x_clk, y_clk, flags, param): # Renamed x,y to x_clk, y_clk
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at: ({x_clk}, {y_clk})") # For help with coordinate calibration
        frame_copy, territories_list, players_list = param  # Unpack params
        # Set debug_ocr=True to generate intermediate image files for OCR
        current_game_state = extract_game_state(frame_copy.copy(), territories_list, players_list, debug_ocr=True)
        print_game_state(current_game_state, players_list)

try:
    ANCHOR_TEMPLATE = cv2.imread("anchor_template.png", cv2.IMREAD_COLOR) # Use COLOR or GRAYSCALE
    if ANCHOR_TEMPLATE is None:
        raise FileNotFoundError
    # If using grayscale template matching:
    # ANCHOR_TEMPLATE_GRAY = cv2.cvtColor(ANCHOR_TEMPLATE, cv2.COLOR_BGR2GRAY)
    AH, AW = ANCHOR_TEMPLATE.shape[:2] # Anchor Height, Width
    print("Anchor template loaded successfully.")
except FileNotFoundError:
    print("ERROR: anchor_template.png not found. Please create it.")
    print("This should be a small, unique, static image from your game map.")
    exit()

                            # # Main execution OLD
                            # if __name__ == "__main__":
                            #     win_name = "War Game Recognition"
                            #     cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE) # AUTOSIZE or NORMAL

                            #     image_path = "war_ss_1.png" # Make sure this path is correct
                            #     frame = cv2.imread(image_path)

                            #     if frame is None:
                            #         print(f"Error: Could not load image from {image_path}")
                            #         exit()
                                
                            #     print(f"Image loaded: {image_path}, Shape: {frame.shape}") # (height, width, channels)

                            #     num_players = 6 # As per the screenshot (Green, Purple, Red, Yellow, Cyan(Blue), Gray)
                                
                            #     players = initialize_players(num_players)
                            #     # Manually set player colors to match screenshot for this test run, if random is not desired for testing:
                            #     # This is useful if you want consistent player color associations during debugging.
                            #     # players[0].player_color = "green"; players[0].name = "green"
                            #     # players[1].player_color = "purple"; players[1].name = "purple"
                            #     # players[2].player_color = "red"; players[2].name = "red"
                            #     # players[3].player_color = "yellow"; players[3].name = "yellow"
                            #     # players[4].player_color = "blue"; players[4].name = "blue" # (Cyan)
                            #     # players[5].player_color = "gray"; players[5].name = "gray"
                            #     # print("Players initialized with specific colors for testing.")
                            #     # for p in players: print(f"Player: {p.name}, Color: {p.player_color}")


                            #     # Cards are not directly used for state extraction from image, but part of game logic
                            #     # cards = create_cards() 

                            #     territories = create_territories(frame.shape)
                                
                            #     # Assign objectives (optional for pure state extraction testing)
                            #     # get_objectives(players)
                            #     # for p in players: print(f"Player {p.name} objective: {p.objective.obj_description if p.objective else 'None'}")


                            #     # Initial game state extraction when script starts
                            #     print("Performing initial game state extraction...")
                            #     initial_game_state = extract_game_state(frame.copy(), territories, players, debug_ocr=True) # Enable OCR debugging
                            #     print_game_state(initial_game_state, players)
                            #     print("\nClick on the image to re-run state extraction with debug info, or press 'q' to quit.")

                            #     cv2.setMouseCallback(win_name, mouse_callback_game_state, (frame, territories, players))
                            #     cv2.imshow(win_name, frame)

                            #     while True:
                            #         key = cv2.waitKey(1) & 0xFF
                            #         if key == ord('q'):
                            #             break
                            #         elif key == ord('r'): # Press 'r' to re-run extraction on the static image
                            #             print("\nRe-running game state extraction...")
                            #             current_game_state = extract_game_state(frame.copy(), territories, players, debug_ocr=True)
                            #             print_game_state(current_game_state, players)


                            #     cv2.destroyAllWindows()

# --- Global variable to signal the main loop ---
trigger_detection = False
app_running = True # To signal the listener thread to stop

# --- Initialize game elements ---
NUM_PLAYERS = 6 # Adjust as needed
PLAYERS = initialize_players(NUM_PLAYERS)
# Important: The coordinates in create_territories MUST be relative to the screen or window you capture.
# If your game is fullscreen at 1600x900, and your coordinates were for that, they might work.
# If your game is in a window, you'll need coordinates relative to the window's top-left.
# Or, capture the full screen and use screen-absolute coordinates.
TERRITORIES = create_territories(None) # Frame shape not known at init for screen capture

def on_press(key):
    global trigger_detection
    try:
        if key.char == 'a':
            print("'a' pressed, triggering detection.")
            trigger_detection = True
    except AttributeError:
        pass # Special keys (like shift, alt, etc.)

def on_release(key):
    global app_running
    if key == keyboard.Key.esc:
        print("Escape pressed, exiting...")
        app_running = False
        return False # Stop listener

def main_loop():
    global trigger_detection
    global app_running

    # Screen capture setup
    # To capture the primary monitor:
    # monitor = {"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    # You need to determine SCREEN_WIDTH and SCREEN_HEIGHT, or capture a specific window.
    # For simplicity, let's try to capture the full primary screen.
    # You might need to adjust 'width' and 'height' if you have multiple monitors or
    # want to capture a specific region.
    # You can find your screen resolution using `xdpyinfo | grep dimensions` in terminal
    
    # --- IMPORTANT: ADJUST THESE MONITOR VALUES ---
    # If your game is full screen 1600x900:
    # screen_width = 1600
    # screen_height = 900
    # If your game is in a window, you need to find its position and size.
    # For now, let's assume full primary screen. You might need to get these dynamically or set them.
    # Example for a 1920x1080 screen:
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    # If your coordinates in create_territories are for a 1600x900 game,
    # and you capture a 1920x1080 screen, the coordinates will be off unless the game
    # is positioned at (0,0) within that 1920x1080 screen.
    # Best practice: Calibrate your territory coordinates based on what `monitor` captures.

    REFERENCE_ANCHOR_POSITION = (1423, 387)

    with mss.mss() as sct:
        # Start the keyboard listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("Listener started. Press 'a' to detect game state. Press 'Esc' to quit.")

        cv2.namedWindow("Live Feed (Optional)", cv2.WINDOW_NORMAL) # Optional window to see what's captured

        while app_running:
            if trigger_detection:
                print("Capturing screen for detection...")
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # --- Find anchor and adjust territory coordinates ---
                # Assuming ANCHOR_TEMPLATE is already loaded (and maybe grayscaled)
                # If ANCHOR_TEMPLATE was loaded as BGR:
                anchor_template_gray_for_match = cv2.cvtColor(ANCHOR_TEMPLATE, cv2.COLOR_BGR2GRAY)
                # If ANCHOR_TEMPLATE was already loaded as grayscale:
                # anchor_template_gray_for_match = ANCHOR_TEMPLATE

                anchor_found = find_anchor_and_adjust_territories(
                    frame_bgr.copy(), # Pass a copy if find_anchor draws on it for debug
                    TERRITORIES, 
                    anchor_template_gray_for_match,
                    REFERENCE_ANCHOR_POSITION 
                )

                if anchor_found:
                    print("Anchor found, positions adjusted.")
                else:
                    print("Anchor NOT found. Using previous/original coordinates.")
                    # Decide behavior: proceed with old coords, skip detection, etc.

                # --- Process the frame using (now potentially adjusted) TERRITORIES ---
                current_game_state = extract_game_state(frame_bgr.copy(), TERRITORIES, PLAYERS, debug_ocr=True)
                print_game_state(current_game_state, PLAYERS)
                
                trigger_detection = False   

                # Optionally display the captured frame (can slow things down)
                cv2.imshow("Live Feed (Optional)", frame_bgr)


            # Add a small delay and check for OpenCV window events if displaying
            if cv2.getWindowProperty("Live Feed (Optional)", cv2.WND_PROP_VISIBLE) >= 1:
                if cv2.waitKey(25) & 0xFF == ord('q'): # Allow quitting via 'q' in OpenCV window too
                    app_running = False
                    break
            else: # If window is closed or not used, just sleep
                time.sleep(0.025) # Small delay to prevent busy-waiting

        print("Stopping listener...")
        listener.stop()
        listener.join() # Wait for listener thread to finish
        cv2.destroyAllWindows()
        print("Application finished.")

if __name__ == "__main__":
    # You might need to set Tesseract command path if it's not in your system PATH
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Or your path

    main_loop()