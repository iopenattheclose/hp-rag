"""Editable Harry Potter lexicon: characters, topics, clearance rules, roles.

This is the ONE file you tweak when tagging looks wrong or when you change the
RBAC design.  build_metadata.py imports it; the RBAC step will import ROLES.

Clearance levels (stored per segment, 0-2):
    0  general narrative / everyday content         (Muggle and up)
    1  spell and potion specifics                   (Professor and up)
    2  Dark Arts, Unforgivable Curses, Horcruxes,
       Deathly Hallows                              (Restricted Section pass only)
A segment's clearance is the MAX over everything it matches.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------------------
# Roles -> which books and what clearance each role may see.
# (Collection-level RBAC: one Qdrant collection per role is built from this.)
# ----------------------------------------------------------------------------
ROLES = {
    "muggle":             {"books": [1, 2, 3, 4, 5, 6, 7], "max_clearance": 0},
    "first_year":         {"books": [1],                   "max_clearance": 1},
    "professor":          {"books": [1, 2, 3, 4, 5, 6, 7], "max_clearance": 1},
    "restricted_section": {"books": [1, 2, 3, 4, 5, 6, 7], "max_clearance": 2},
}

# ----------------------------------------------------------------------------
# Characters: canonical name -> aliases (matched as whole words, case-sensitive).
# Heuristic on purpose: "Potter" can also be James/Lily; "Malfoy" can be Lucius.
# ----------------------------------------------------------------------------
CHARACTERS: dict[str, list[str]] = {
    "Harry Potter": ["Harry", "Potter"],
    "Ron Weasley": ["Ron", "Ronald"],
    "Hermione Granger": ["Hermione", "Granger"],
    "Albus Dumbledore": ["Dumbledore", "Albus"],
    "Severus Snape": ["Snape", "Severus"],
    "Lord Voldemort": ["Voldemort", "You-Know-Who", "He-Who-Must-Not-Be-Named",
                       "Tom Riddle", "Riddle", "the Dark Lord", "Dark Lord"],
    "Rubeus Hagrid": ["Hagrid", "Rubeus"],
    "Minerva McGonagall": ["McGonagall", "Minerva"],
    "Draco Malfoy": ["Draco", "Malfoy"],
    "Lucius Malfoy": ["Lucius"],
    "Narcissa Malfoy": ["Narcissa"],
    "Neville Longbottom": ["Neville", "Longbottom"],
    "Quirinus Quirrell": ["Quirrell"],
    "Sirius Black": ["Sirius", "Padfoot"],
    "Remus Lupin": ["Lupin", "Remus", "Moony"],
    "Peter Pettigrew": ["Pettigrew", "Wormtail", "Scabbers"],
    "Bellatrix Lestrange": ["Bellatrix", "Lestrange"],
    "Dobby": ["Dobby"],
    "Kreacher": ["Kreacher"],
    "Ginny Weasley": ["Ginny"],
    "Fred Weasley": ["Fred"],
    "George Weasley": ["George"],
    "Percy Weasley": ["Percy"],
    "Bill Weasley": ["Bill"],
    "Molly Weasley": ["Mrs. Weasley", "Molly"],
    "Arthur Weasley": ["Mr. Weasley", "Arthur"],
    "Luna Lovegood": ["Luna", "Lovegood"],
    "Cedric Diggory": ["Cedric", "Diggory"],
    "Cho Chang": ["Cho"],
    "Viktor Krum": ["Krum", "Viktor"],
    "Fleur Delacour": ["Fleur", "Delacour"],
    "Alastor Moody": ["Moody", "Mad-Eye"],
    "Dolores Umbridge": ["Umbridge", "Dolores"],
    "Horace Slughorn": ["Slughorn"],
    "Gilderoy Lockhart": ["Lockhart", "Gilderoy"],
    "Sybill Trelawney": ["Trelawney"],
    "Argus Filch": ["Filch"],
    "Dudley Dursley": ["Dudley"],
    "Vernon Dursley": ["Uncle Vernon", "Vernon"],
    "Petunia Dursley": ["Aunt Petunia", "Petunia"],
    "James Potter": ["James"],
    "Lily Potter": ["Lily"],
    "Peeves": ["Peeves"],
    "Nearly Headless Nick": ["Nearly Headless Nick", "Sir Nicholas"],
    "Moaning Myrtle": ["Myrtle"],
    "Fawkes": ["Fawkes"],
    "Buckbeak": ["Buckbeak", "Witherwings"],
    "Hedwig": ["Hedwig"],
    "Fluffy": ["Fluffy"],
    "Aragog": ["Aragog"],
    "Nagini": ["Nagini"],
    "Kingsley Shacklebolt": ["Kingsley", "Shacklebolt"],
    "Nymphadora Tonks": ["Tonks"],
    "Rita Skeeter": ["Skeeter"],
    "Griphook": ["Griphook"],
    "Ollivander": ["Ollivander"],
    "Seamus Finnigan": ["Seamus"],
    "Dean Thomas": ["Dean Thomas"],
    "Lavender Brown": ["Lavender"],
    "Oliver Wood": ["Oliver Wood"],
    "Gellert Grindelwald": ["Grindelwald"],
    "Aberforth Dumbledore": ["Aberforth"],
    "Rufus Scrimgeour": ["Scrimgeour"],
    "Cornelius Fudge": ["Fudge"],
    "Mundungus Fletcher": ["Mundungus"],
    "Xenophilius Lovegood": ["Xenophilius"],
    "Crabbe": ["Crabbe"],
    "Goyle": ["Goyle"],
    "Pansy Parkinson": ["Pansy"],
    "Marcus Flint": ["Flint"],
    "Madam Pomfrey": ["Pomfrey"],
    "Madam Hooch": ["Hooch"],
    "Pomona Sprout": ["Sprout"],
    "Filius Flitwick": ["Flitwick"],
    "Firenze": ["Firenze"],
    "Winky": ["Winky"],
}

# ----------------------------------------------------------------------------
# Topics: topic -> terms (whole-word, case-insensitive, optional plural -s/-es).
# TOPIC_CLEARANCE says how restricted a topic is; topics not listed are 0.
# ----------------------------------------------------------------------------
TOPICS: dict[str, list[str]] = {
    "spells": [
        "spell", "charm", "jinx", "hex", "countercurse", "counter-curse",
        "incantation", "Expelliarmus", "Lumos", "Nox", "Accio",
        "Wingardium Leviosa", "Alohomora", "Stupefy", "Petrificus Totalus",
        "Expecto Patronum", "Riddikulus", "Obliviate", "Reparo", "Incendio",
        "Protego", "Sectumsempra", "Confundo", "Impedimenta", "Reducto",
        "Diffindo", "Engorgio", "Reducio", "Levicorpus", "Muffliato",
        "Episkey", "Silencio", "Rictusempra", "Locomotor", "Finite Incantatem",
    ],
    "potions": [
        "potion", "Polyjuice", "Veritaserum", "Felix Felicis", "Amortentia",
        "Wolfsbane", "Draught of Living Death", "Skele-Gro", "Pepperup",
        "Wiggenweld", "antidote", "elixir", "Draught of Peace",
    ],
    "magic_items": [
        "wand", "Invisibility Cloak", "Portkey", "Time-Turner", "Remembrall",
        "Marauder's Map", "Pensieve", "Deluminator", "Sneakoscope",
        "broomstick", "Firebolt", "Nimbus Two Thousand", "Floo powder",
    ],
    "creatures": [
        "dragon", "hippogriff", "basilisk", "dementor", "centaur", "thestral",
        "troll", "werewolf", "boggart", "giant", "goblin", "house-elf",
        "phoenix", "unicorn", "Acromantula", "grindylow", "Blast-Ended Skrewt",
        "three-headed dog", "merpeople", "veela", "Cornish pixie",
    ],
    "quidditch": [
        "Quidditch", "Snitch", "Quaffle", "Bludger", "Seeker", "Beater", "Chaser",
    ],
    "hogwarts_lore": [
        "Sorting Hat", "Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw",
        "common room", "house points", "Great Hall", "Forbidden Forest",
        "Room of Requirement", "Chamber of Secrets", "Hogsmeade",
        "Diagon Alley", "Knockturn Alley", "Gringotts", "Hogwarts Express",
    ],
    "ministry_order": [
        "Ministry of Magic", "Auror", "Azkaban", "Order of the Phoenix",
        "Dumbledore's Army", "Triwizard", "Wizengamot", "Daily Prophet",
    ],
    # --- restricted material ------------------------------------------------
    "unforgivable_curses": [
        "Unforgivable", "Avada Kedavra", "killing curse", "Cruciatus", "Crucio",
        "Imperius", "Imperio",
    ],
    "dark_arts": ["Dark Arts", "dark magic"],          # general mentions -> level 1
    "death_eaters_dark_mark": ["Dark Mark", "Death Eater", "Inferius", "Inferi"],
    # incantations / effects of Dark spells (policy A: technique knowledge is restricted)
    "dark_spells": ["Fiendfyre", "cursed fire", "Sectumsempra", "Morsmordre"],
    "horcruxes": ["Horcrux"],
    "hallows": [
        "Deathly Hallows", "Elder Wand", "Resurrection Stone", "Peverell",
    ],
}

TOPIC_CLEARANCE: dict[str, int] = {
    "spells": 1,
    "potions": 1,
    "unforgivable_curses": 2,
    "dark_arts": 1,
    "death_eaters_dark_mark": 1,   # plot-level in books 4-7; was 2, over-restricted (371 segments)
    "dark_spells": 2,
    "horcruxes": 2,
    "hallows": 2,
}

# Gap fill: a short run of lower-clearance segments sandwiched (same scene) between
# two segments of level L is part of the same restricted passage -> raise it to L.
# e.g. the Fiendfyre scene: s042=2, s043=0, s044=0, s045=2  ->  s043, s044 become 2.
GAP_FILL = {"levels": [2], "max_gap": 2}


# Text removed before matching a topic (e.g. the school CLASS is not restricted).
TOPIC_EXCLUDE: dict[str, re.Pattern] = {
    "dark_arts": re.compile(r"Defen[cs]e\s+Against\s+the\s+Dark\s+Arts", re.I),
}


# ----------------------------------------------------------------------------
# Compiled matchers (built once on import)
# ----------------------------------------------------------------------------
def _alt(terms: list[str]) -> str:
    return "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))


CHAR_RES = {
    name: re.compile(r"(?<![\w-])(?:" + _alt(aliases) + r")(?![\w])")
    for name, aliases in CHARACTERS.items()
}
TOPIC_RES = {
    topic: re.compile(r"\b(?:" + _alt(terms) + r")(?:s|es)?\b", re.IGNORECASE)
    for topic, terms in TOPICS.items()
}