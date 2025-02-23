#!/usr/bin/env python3
"""
VRN Generator

Generates UK vehicle registration numbers (plates) according to the following rules:
  • Format: 2 letters (memory tag) + 2 numbers (age identifier) + space + 3 random letters.
  • Memory tag (first two letters) is chosen from groups as follows:
      – 25% chance from those starting with L (London: LA–LY)
      – 25% chance from those starting with B (Birmingham: BA–BY)
      – 25% chance from those starting with M (Manchester/Merseyside: MA–MY)
      – 25% chance from all the remaining allowed pairs.
    The allowed memory tag pairs are generated from a fixed set of allowed characters.
   (Note some groups—for V we only allow "VA" and "VY".)
  • Age identifier (two digits) is chosen from all allowed values:
      Allowed values: 02–14, 15–25, 52–64, 65–75.
    Weight 75% of the time toward the "most common" groups: 15–25 and 65–75; 25% from the rest.
  • The trailing three letters are chosen completely at random (from A–Z).
  • In addition, the plate may show an optional "flash" (a colored rectangle on the left)
    that sometimes carries a national flag and/or national identifier text.
    Flash colour is chosen weighted as follows:
      – blue (70%), green (20%), or no flash (NONE) (10%).
    If a flash is present then there is a 50% chance to include the national identifier data.
    When included, the national identifier is chosen (weighted) as:
      – UK (80%), ENG (10%), SCO (10%).
    For the flag itself:
      – If national id = UK then the Union Jack is added 50% of the time.
      – If national id = ENG then with 90% chance "Cross of St George" (ENG) is added.
      – If national id = SCO then with 90% chance "Cross of St Andrew" (SCO) is added.
          
The CSV output has four columns: VRN,flash,flag,country
Example rows:
    MH73 XWN,blue,UK,UK
    LB21 DRO,green,NONE,NONE
    LA18 ASF,NONE,NONE,NONE
    M20 AKH,blue,NONE,UK
"""

import csv
import random
import string
import sys

# COUNT determines how many VRNs to generate; you may override this
COUNT = 50

# Allowed memory tag list exactly as in the snippet.
ALLOWED_MEMORY_TAGS = [
    "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AR", "AS", "AT", "AU", "AV", "AW", "AX", "AY",
    "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV", "BW", "BX", "BY",
    "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH", "CJ", "CK", "CL", "CM", "CN", "CO", "CP", "CR", "CS", "CT", "CU", "CV", "CW", "CX", "CY",
    "DA", "DB", "DC", "DD", "DE", "DF", "DG", "DH", "DJ", "DK", "DL", "DM", "DN", "DO", "DP", "DR", "DS", "DT", "DU", "DV", "DW", "DX", "DY",
    "EA", "EB", "EC", "ED", "EE", "EF", "EG", "EH", "EI", "EJ", "EK", "EL", "EM", "EN", "EO", "EP", "EQ", "ER", "ES", "ET", "EU", "EV", "EW", "EX", "EY",
    "FA", "FB", "FC", "FD", "FE", "FF", "FG", "FH", "FJ", "FK", "FL", "FM", "FN", "FP", "FR", "FS", "FT", "FV", "FW", "FX", "FY",
    "GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH", "GJ", "GK", "GL", "GM", "GN", "GO", "GP", "GR", "GS", "GT", "GU", "GV", "GW", "GX", "GY",
    "HA", "HB", "HC", "HD", "HE", "HF", "HG", "HH", "HJ", "HK", "HL", "HM", "HN", "HO", "HP", "HR", "HS", "HT", "HU", "HV", "HW", "HX", "HY",
    "KA", "KB", "KC", "KD", "KE", "KF", "KG", "KH", "KJ", "KK", "KL", "KM", "KN", "KO", "KP", "KR", "KS", "KT", "KU", "KV", "KW", "KX", "KY",
    "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH", "LJ", "LK", "LL", "LM", "LN", "LO", "LP", "LR", "LS", "LT", "LU", "LV", "LW", "LX", "LY",
    "MA", "MB", "MC", "MD", "ME", "MF", "MG", "MH", "MI", "MJ", "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY",
    "NA", "NB", "NC", "ND", "NE", "NG", "NH", "NJ", "NK", "NL", "NM", "NN", "NO", "NP", "NR", "NS", "NT", "NU", "NV", "NW", "NX", "NY",
    "OA", "OB", "OC", "OD", "OE", "OF", "OG", "OH", "OI", "OJ", "OK", "OL", "OM", "ON", "OO", "OP", "OQ", "OR", "OS", "OT", "OU", "OV", "OW", "OX", "OY",
    "PA", "PB", "PC", "PD", "PE", "PF", "PG", "PH", "PJ", "PK", "PL", "PM", "PN", "PO", "PP", "PR", "PS", "PT", "PU", "PV", "PW", "PX", "PY",
    "RA", "RB", "RC", "RD", "RE", "RF", "RG", "RH", "RI", "RJ", "RK", "RL", "RM", "RN", "RO", "RP", "RQ", "RR", "RS", "RT", "RU", "RV", "RW", "RX", "RY",
    "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SJ", "SK", "SL", "SM", "SN", "SO", "SP", "SR", "SS", "ST", "SU", "SV", "SW", "SX", "SY",
    "VA", "VB", "VC", "VD", "VE", "VF", "VG", "VH", "VI", "VJ", "VK", "VL", "VM", "VN", "VO", "VP", "VQ", "VR", "VS", "VT", "VU", "VV", "VW", "VX", "VY",
    "WA", "WB", "WC", "WD", "WE", "WF", "WG", "WH", "WJ", "WK", "WL", "WM", "WN", "WO", "WP", "WR", "WS", "WT", "WU", "WV", "WW", "WX", "WY",
    "YA", "YB", "YC", "YD", "YE", "YF", "YG", "YH", "YJ", "YK", "YL", "YM", "YN", "YO", "YP", "YR", "YS", "YT", "YU", "YV", "YW", "YX", "YY"
]

def build_memory_tags():
    """
    Return the complete list of allowed memory tag pairs as specified.
    """
    return ALLOWED_MEMORY_TAGS

def build_memory_tag_groups(all_tags):
    """
    Partition allowed memory tags into groups:
        - group_L: tags starting with L (London)
        - group_B: tags starting with B (Birmingham)
        - group_M: tags starting with M (Manchester/Merseyside)
        - group_rest: all remaining tags (other regions)
    """
    group_L = [tag for tag in all_tags if tag.startswith("L")]
    group_B = [tag for tag in all_tags if tag.startswith("B")]
    group_M = [tag for tag in all_tags if tag.startswith("M")]
    group_rest = [tag for tag in all_tags if tag[0] not in ["L", "B", "M"]]
    return group_L, group_B, group_M, group_rest

def choose_memory_tag():
    """
    Choose a memory tag (2 letters) based on the following weighting:
      - 25% chance from group_L, 25% from group_B, 25% from group_M, and 25% from group_rest.
    """
    all_tags = build_memory_tags()
    group_L, group_B, group_M, group_rest = build_memory_tag_groups(all_tags)
    group_choice = random.choices(
        population=["L", "B", "M", "rest"],
        weights=[25, 25, 25, 25],
        k=1
    )[0]
    if group_choice == "L":
        tag = random.choice(group_L)
    elif group_choice == "B":
        tag = random.choice(group_B)
    elif group_choice == "M":
        tag = random.choice(group_M)
    else:
        tag = random.choice(group_rest)
    return tag

def choose_age_identifier():
    """
    Choose the two-digit age identifier.
    Allowed values:
       Group A (75% chance): 15–25 and 65–75
       Group B (25% chance): 02–14 and 52–64
    Returns a string formatted with two digits.
    """
    groupA = list(range(15, 26)) + list(range(65, 76))
    groupB = list(range(2, 15)) + list(range(52, 65))
    if random.random() < 0.75:
        num = random.choice(groupA)
    else:
        num = random.choice(groupB)
    return f"{num:02d}"

def choose_random_letters():
    """
    Return 3 random uppercase letters (for the trailing part).
    """
    return "".join(random.choices(string.ascii_uppercase, k=3))

def choose_vrn():
    """
    Build one VRN string (without flash options).
    Format: "<memory tag><age identifier> <3 random letters>"
    """
    memory_tag = choose_memory_tag()
    age_id = choose_age_identifier()
    letters = choose_random_letters()
    return f"{memory_tag}{age_id} {letters}"

def choose_flash_data():
    """
    Decide on the optional flash strip settings (flash colour, flag and country):
      - Flash colour is chosen as:
           blue: 70%, green: 20%, NONE: 10%
      - If flash colour is NONE then no additional national identifier is printed.
      - Otherwise (if flash is blue or green), there is a 50% chance to include national identifier data.
      - When included, select the national identifier code from:
           UK (80%), ENG (10%), SCO (10%)
        and then decide on the optional flag:
           * If national id == UK, include the Union Jack flag with 50% chance.
           * If national id == ENG, with 90% chance "Cross of St George" (ENG) is added.
           * If national id == SCO, with 90% chance "Cross of St Andrew" (SCO) is added.
    Returns a tuple (flash, flag, country) where flash is the colour string (or "NONE").
    """
    # Choose flash colour.
    flash = random.choices(
        population=["blue", "green", "NONE"],
        weights=[70, 20, 10],
        k=1
    )[0]
    # By default, no national identifier data.
    flag = "NONE"
    country = "NONE"
    # If a flash strip exists (i.e. flash is not "NONE"), then decide if we include national identifier data.
    if flash != "NONE":
        if random.random() < 0.5:
            # Include national identifier.
            country = random.choices(
                population=["UK", "ENG", "SCO"],
                weights=[80, 10, 10],
                k=1
            )[0]
            # Decide on flag according to national identifier.
            if country == "UK":
                # Add Union Jack 50% of the time.
                flag = "UK" if random.random() < 0.5 else "NONE"
            elif country == "ENG":
                # 90% chance to include English flag
                flag = "ENG" if random.random() < 0.9 else "NONE"
            elif country == "SCO":
                # 90% chance to include Scottish flag
                flag = "SCO" if random.random() < 0.9 else "NONE"
    return flash, flag, country

def generate_plate_record():
    """
    Generate one complete plate record.
    Returns a dict with keys: VRN, flash, flag, country
    """
    vrn = choose_vrn()
    flash, flag, country = choose_flash_data()
    return {
        'VRN': vrn,
        'flash': flash,
        'flag': flag,
        'country': country
    }

def generate_plates(count=COUNT):
    """
    Generate multiple plate records ensuring no duplicate VRNs.

    Args:
        count (int): Number of plates to generate
    Returns:
        list: List of dicts, each containing a unique plate record data.
    """
    records = []
    seen_vrns = set()
    while len(records) < count:
        record = generate_plate_record()
        if record['VRN'] not in seen_vrns:
            records.append(record)
            seen_vrns.add(record['VRN'])
    return records

def main():
    count = COUNT
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass

    # Generate records
    records = generate_plates(count)
    
    # Write CSV to stdout
    writer = csv.DictWriter(sys.stdout, fieldnames=['VRN', 'flash', 'flag', 'country'])
    writer.writeheader()
    writer.writerows(records)

if __name__ == "__main__":
    main()