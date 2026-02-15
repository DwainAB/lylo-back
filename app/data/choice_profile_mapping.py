# Mapping entre les choix du questionnaire et les profils olfactifs.
# Chaque choix est associé à 2 profils. Le premier est le match principal.

CHOICE_PROFILE_MAPPING = {
    1: {  # Quelle destination vous attire le plus ?
        "Ville": ["Influencer", "Visionary"],
        "Fôret": ["Strategist", "Creator"],
        "Campagne": ["Cosy", "Icon"],
        "Montagne": ["Innovator", "Trailblazer"],
        "Désert": ["Disruptor", "Visionary"],
        "Plage": ["Trailblazer", "Influencer"],
    },
    2: {  # Quelle ville préfériez-vous visiter ?
        "New york": ["Influencer", "Innovator"],
        "Athène": ["Icon", "Strategist"],
        "Delhi": ["Creator", "Trailblazer"],
        "Monsanto": ["Cosy", "Strategist"],
        "Pekin": ["Visionary", "Disruptor"],
        "Tombouctou": ["Disruptor", "Trailblazer"],
    },
    3: {  # Quelle genre de restaurant préférez-vous ?
        "Gastronomique": ["Icon", "Strategist"],
        "Salade": ["Trailblazer", "Cosy"],
        "Couscous": ["Creator", "Trailblazer"],
        "Cuisine Moléculaire": ["Disruptor", "Visionary"],
        "Steak House": ["Innovator", "Disruptor"],
        "Déssert": ["Influencer", "Cosy"],
    },
    4: {  # Quelle activité vous attire le plus ?
        "Sport": ["Innovator", "Trailblazer"],
        "Rencontre population": ["Creator", "Influencer"],
        "Promenade découverte": ["Visionary", "Strategist"],
        "Musée et art": ["Icon", "Creator"],
        "musique, bar, club et casino": ["Disruptor", "Influencer"],
        "lecture, film et plage": ["Cosy", "Strategist"],
    },
    5: {  # Quelle sport préférez-vous ?
        "Endurance": ["Trailblazer", "Innovator"],
        "équipe": ["Influencer", "Creator"],
        "Précision": ["Strategist", "Visionary"],
        "Aventure": ["Disruptor", "Trailblazer"],
        "Musculation": ["Innovator", "Disruptor"],
        "Agrément": ["Cosy", "Icon"],
    },
    6: {  # Quel musique vous inspire le plus ?
        "Classique": ["Icon", "Strategist"],
        "Lyrique et choral": ["Icon", "Creator"],
        "Electonique et moderne": ["Disruptor", "Visionary"],
        "Jazz et new edge": ["Visionary", "Strategist"],
        "Son d'ailleurs": ["Creator", "Trailblazer"],
        "Rock": ["Innovator", "Disruptor"],
    },
    7: {  # Quelle matière vous attire le plus ?
        "Soie": ["Icon", "Influencer"],
        "Lin": ["Trailblazer", "Cosy"],
        "Neige": ["Visionary", "Trailblazer"],
        "Pierre": ["Strategist", "Disruptor"],
        "Bois": ["Innovator", "Strategist"],
        "Ambre chaud": ["Creator", "Cosy"],
    },
    8: {  # Quelle plante aimez-vous le plus ?
        "Fleurs": ["Influencer", "Icon"],
        "Plante aromatique": ["Strategist", "Trailblazer"],
        "Plante sauvage": ["Disruptor", "Creator"],
        "Patchiouli et vetiver": ["Visionary", "Innovator"],
        "Mousse sous bois": ["Cosy", "Strategist"],
        "Foin au soleil": ["Trailblazer", "Cosy"],
    },
    9: {  # Quel type de parfum préférez-vous ?
        "Léger": ["Cosy", "Trailblazer"],
        "Caractère": ["Disruptor", "Innovator"],
        "Vintage": ["Icon", "Strategist"],
        "Chaud": ["Creator", "Strategist"],
        "Frais": ["Trailblazer", "Influencer"],
        "Moderne": ["Visionary", "Disruptor"],
    },
    10: {  # Quelle matière première vous attire le plus ?
        "Musc": ["Cosy", "Trailblazer"],
        "Ambre": ["Icon", "Creator"],
        "Bois": ["Innovator", "Strategist"],
        "Gourmand": ["Influencer", "Cosy"],
        "Floral": ["Influencer", "Creator"],
        "Marin": ["Trailblazer", "Disruptor"],
    },
    11: {  # Quelle couleurs préférez-vous ?
        "Marron et beige": ["Strategist", "Cosy"],
        "Vert": ["Trailblazer", "Innovator"],
        "Rouge": ["Creator", "Disruptor"],
        "Bleu": ["Visionary", "Trailblazer"],
        "Jaune": ["Influencer", "Creator"],
        "Noir": ["Icon", "Disruptor"],
    },
    12: {  # Quel type d'art vous attire le plus ?
        "Comprendre": ["Strategist", "Visionary"],
        "Reconnaitre": ["Icon", "Cosy"],
        "Immaginer": ["Creator", "Visionary"],
        "s'évader": ["Trailblazer", "Disruptor"],
        "s'intérroger": ["Innovator", "Strategist"],
        "s'émmerveiller": ["Influencer", "Cosy"],
    },
}

# Genre associé à chaque profil (pour le bonus de scoring)
PROFILE_GENDERS = {
    "Disruptor": "masculine",
    "Visionary": "unisex",
    "Trailblazer": "unisex",
    "Strategist": "masculine",
    "Innovator": "masculine",
    "Creator": "feminine",
    "Influencer": "feminine",
    "Icon": "feminine",
    "Cosy": "unisex",
}

# Descriptions des profils (pour la présentation des formules)
PROFILE_DESCRIPTIONS = {
    "Disruptor": "Innovant et atypique. Pas l'approche standard, il trace de nouveaux chemins à travers les anciens.",
    "Visionary": "Toujours plusieurs coups d'avance, avec une direction claire et de la détermination.",
    "Trailblazer": "Frais, lumineux, prêt pour tout. Un mélange subtil de floral, d'agrumes et de notes profondes.",
    "Strategist": "Réfléchi, patient, jamais pressé. Une présence apaisante et assurée.",
    "Innovator": "Jamais le même chemin deux fois, audacieux et risqué, mais assuré.",
    "Creator": "Joyeux, créatif, excentrique, pleinement soi-même, sans compromis.",
    "Influencer": "Toujours à la recherche de la nouveauté, enthousiaste et confiant.",
    "Icon": "Au-delà du classique, sage et toujours à la mode. Intemporel, sans effort et élégant.",
    "Cosy": "Doux, discret, rassuré par le confort. Une bulle de bien-être.",
}

PROFILE_DESCRIPTIONS_EN = {
    "Disruptor": "Innovative and unconventional. Not the standard approach, they blaze new trails through old paths.",
    "Visionary": "Always several moves ahead, with a clear direction and determination.",
    "Trailblazer": "Fresh, bright, ready for anything. A subtle blend of floral, citrus and deep notes.",
    "Strategist": "Thoughtful, patient, never in a rush. A calming and confident presence.",
    "Innovator": "Never the same path twice, bold and daring, yet assured.",
    "Creator": "Joyful, creative, eccentric, fully themselves, without compromise.",
    "Influencer": "Always seeking novelty, enthusiastic and confident.",
    "Icon": "Beyond classic, wise and always fashionable. Timeless, effortless and elegant.",
    "Cosy": "Soft, discreet, comforted by coziness. A bubble of well-being.",
}

# Traduction anglais → français des noms d'ingrédients (notes olfactives)
INGREDIENT_EN_TO_FR = {
    # TOP NOTES
    "Grapefruit wood": "Bois de pamplemousse",
    "Bamboo leaf": "Feuille de bambou",
    "Warm spices": "Épices chaudes",
    "Cold spices": "Épices froides",
    "Blackcurrant": "Cassis",
    "White dew": "Rosée blanche",
    "Grasse's lavender": "Lavande de Grasse",
    "Fresh bergamot": "Bergamote fraîche",
    "Modern freshness": "Fraîcheur moderne",
    "Wild rose": "Rose sauvage",
    # HEART NOTES
    "Rose & peony": "Rose & pivoine",
    "Green lily of the valley": "Muguet vert",
    "Jasmine blossom": "Fleur de jasmin",
    "Fresh breeze": "Brise fraîche",
    "Tutti fruitti-rhubarb": "Tutti frutti-rhubarbe",
    "Neroli-orange blossom": "Néroli-fleur d'oranger",
    "Fig tree": "Figuier",
    "Powdery violet": "Violette poudrée",
    "Floral honey": "Miel floral",
    "Blond tobacco": "Tabac blond",
    # BASE NOTES
    "Sweet amber": "Ambre sucré",
    "Iris powder": "Poudre d'iris",
    "Sweet note": "Note sucrée",
    "Woody bouquet": "Bouquet boisé",
    "Woody harmony": "Harmonie boisée",
    "White chypre": "Chypre blanc",
    "Aldehyde harmony": "Harmonie aldéhydée",
    "Powdery musk": "Musc poudré",
    "Modern wood": "Bois moderne",
    "Leather": "Cuir",
}
