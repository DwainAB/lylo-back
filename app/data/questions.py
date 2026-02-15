QUESTIONS_EN = [
    {
        "id": 1,
        "question": "Which destination attracts you the most?",
        "choices": ["City", "Forest", "Countryside", "Mountain", "Desert", "Beach"],
    },
    {
        "id": 2,
        "question": "Which city would you prefer to visit?",
        "choices": ["New York", "Athens", "Delhi", "Monsanto", "Beijing", "Timbuktu"],
    },
    {
        "id": 3,
        "question": "What kind of restaurant do you prefer?",
        "choices": ["Gourmet", "Salad", "Couscous", "Molecular Cuisine", "Steak House", "Dessert"],
    },
    {
        "id": 4,
        "question": "Which activity attracts you the most?",
        "choices": ["Sport", "Meeting locals", "Discovery walk", "Museum and art", "Music, bar, club and casino", "Reading, movies and beach"],
    },
    {
        "id": 5,
        "question": "Which sport do you prefer?",
        "choices": ["Endurance", "Team", "Precision", "Adventure", "Weightlifting", "Leisure"],
    },
    {
        "id": 6,
        "question": "Which music inspires you the most?",
        "choices": ["Classical", "Lyrical and choral", "Electronic and modern", "Jazz and new age", "World music", "Rock"],
    },
    {
        "id": 7,
        "question": "Which material attracts you the most?",
        "choices": ["Silk", "Linen", "Snow", "Stone", "Wood", "Warm amber"],
    },
    {
        "id": 8,
        "question": "Which plant do you like the most?",
        "choices": ["Flowers", "Aromatic plants", "Wild plants", "Patchouli and vetiver", "Forest moss", "Hay in the sun"],
    },
    {
        "id": 9,
        "question": "What type of perfume do you prefer?",
        "choices": ["Light", "Bold", "Vintage", "Warm", "Fresh", "Modern"],
    },
    {
        "id": 10,
        "question": "Which raw material attracts you the most?",
        "choices": ["Musk", "Amber", "Wood", "Gourmand", "Floral", "Marine"],
    },
    {
        "id": 11,
        "question": "Which colors do you prefer?",
        "choices": ["Brown and beige", "Green", "Red", "Blue", "Yellow", "Black"],
    },
    {
        "id": 12,
        "question": "What type of art attracts you the most?",
        "choices": ["Understanding", "Recognizing", "Imagining", "Escaping", "Questioning", "Wondering"],
    },
]

# Mapping des choix anglais vers les clés françaises (utilisées par le scoring)
EN_TO_FR_CHOICES = {
    1: {"City": "Ville", "Forest": "Fôret", "Countryside": "Campagne", "Mountain": "Montagne", "Desert": "Désert", "Beach": "Plage"},
    2: {"New York": "New york", "Athens": "Athène", "Delhi": "Delhi", "Monsanto": "Monsanto", "Beijing": "Pekin", "Timbuktu": "Tombouctou"},
    3: {"Gourmet": "Gastronomique", "Salad": "Salade", "Couscous": "Couscous", "Molecular Cuisine": "Cuisine Moléculaire", "Steak House": "Steak House", "Dessert": "Déssert"},
    4: {"Sport": "Sport", "Meeting locals": "Rencontre population", "Discovery walk": "Promenade découverte", "Museum and art": "Musée et art", "Music, bar, club and casino": "musique, bar, club et casino", "Reading, movies and beach": "lecture, film et plage"},
    5: {"Endurance": "Endurance", "Team": "équipe", "Precision": "Précision", "Adventure": "Aventure", "Weightlifting": "Musculation", "Leisure": "Agrément"},
    6: {"Classical": "Classique", "Lyrical and choral": "Lyrique et choral", "Electronic and modern": "Electonique et moderne", "Jazz and new age": "Jazz et new edge", "World music": "Son d'ailleurs", "Rock": "Rock"},
    7: {"Silk": "Soie", "Linen": "Lin", "Snow": "Neige", "Stone": "Pierre", "Wood": "Bois", "Warm amber": "Ambre chaud"},
    8: {"Flowers": "Fleurs", "Aromatic plants": "Plante aromatique", "Wild plants": "Plante sauvage", "Patchouli and vetiver": "Patchiouli et vetiver", "Forest moss": "Mousse sous bois", "Hay in the sun": "Foin au soleil"},
    9: {"Light": "Léger", "Bold": "Caractère", "Vintage": "Vintage", "Warm": "Chaud", "Fresh": "Frais", "Modern": "Moderne"},
    10: {"Musk": "Musc", "Amber": "Ambre", "Wood": "Bois", "Gourmand": "Gourmand", "Floral": "Floral", "Marine": "Marin"},
    11: {"Brown and beige": "Marron et beige", "Green": "Vert", "Red": "Rouge", "Blue": "Bleu", "Yellow": "Jaune", "Black": "Noir"},
    12: {"Understanding": "Comprendre", "Recognizing": "Reconnaitre", "Imagining": "Immaginer", "Escaping": "s'évader", "Questioning": "s'intérroger", "Wondering": "s'émmerveiller"},
}

# Mapping choix → fichier image (par question_id, indexé par position du choix)
CHOICE_IMAGES = {
    1: ["city.avif", "wood.webp", "campagne.webp", "montagne.webp", "desert.jpg", "plage.webp"],
    2: ["newyork.webp", "athene.jpg", "delhi.jpg", "monsanto.webp", "pekin.jpg", "tombouctou.jpg"],
    3: ["gastro.webp", "salade.jpg", "couscous.jpeg", "moleculaire.jpg", "steakhouse.jpg", "dessert.webp"],
    4: ["sport.webp", "rencontre-population.jpg", "promenade-decouverte.jpg", "musée-art.jpg", "musique-bar-club.jpeg", "lecture-film.jpg"],
    5: ["endurence.webp", "equipe.jpg", "precision.jpg", "aventure.jpeg", "musculation.jpg", "agrement.jpg"],
    6: ["classique.jpg", "lyrique-chorale.jpg", "electronique-moderne.webp", "jazz-newedge.webp", "son-ailleur.jpeg", "rock.jpg"],
    7: ["soie.jpg", "lin.jpg", "neige.jpg", "pierre.jpg", "bois.jpg", "ambre.jpg"],
    8: ["fleurs.png", "plante-aromatiques.jpg", "plante-sauvage.webp", "PATCHOULI : VETIVER.webp", "mousse.jpg", "foin.png"],
    9: ["leger.webp", "caractere.jpg", "vintage.jpeg", "chaud.webp", "frais.jpg", "moderne.webp"],
    10: ["musc.jpg", "ambre.webp", "bois.gif", "gourmant.webp", "floral.jpg", "marin.jpg"],
    11: ["marron-beige.jpg", "vert.jpg", "rouge.jpg", "bleu.webp", "jaune.jpg", "noir.jpg"],
    12: ["comprendre.jpg", "reconnaitre.png", "imaginer.jpg", "evader.jpg", "interroger.png", "emerveiller.jpg"],
}


def _enrich_questions(questions: list) -> list:
    """Ajoute l'URL d'image à chaque choix."""
    enriched = []
    for q in questions:
        images = CHOICE_IMAGES.get(q["id"], [])
        choices_with_images = []
        for i, choice in enumerate(q["choices"]):
            entry = {"label": choice}
            if i < len(images):
                entry["image"] = f"/static/choices/{q['id']}/{images[i]}"
            choices_with_images.append(entry)
        enriched.append({
            **q,
            "choices": choices_with_images,
        })
    return enriched


QUESTIONS_FR = [
    {
        "id": 1,
        "question": "Quelle destination vous attire le plus ?",
        "choices": ["Ville", "Fôret", "Campagne", "Montagne", "Désert", "Plage"],
    },
    {
        "id": 2,
        "question": "Quelle ville préfériez-vous visiter ?",
        "choices": ["New york", "Athène", "Delhi", "Monsanto", "Pekin", "Tombouctou"],
    },
    {
        "id": 3,
        "question": "Quelle genre de restaurant préférez-vous ?",
        "choices": ["Gastronomique", "Salade", "Couscous", "Cuisine Moléculaire", "Steak House", "Déssert"],
    },
    {
        "id": 4,
        "question": "Quelle activité vous attire le plus ?",
        "choices": ["Sport", "Rencontre population", "Promenade découverte", "Musée et art", "musique, bar, club et casino", "lecture, film et plage"],
    },
    {
        "id": 5,
        "question": "Quelle sport préférez-vous ?",
        "choices": ["Endurance", "équipe", "Précision", "Aventure", "Musculation", "Agrément"],
    },
    {
        "id": 6,
        "question": "Quel musique vous inspire le plus ?",
        "choices": ["Classique", "Lyrique et choral", "Electonique et moderne", "Jazz et new edge", "Son d'ailleurs", "Rock"],
    },
    {
        "id": 7,
        "question": "Quelle matière vous attire le plus ?",
        "choices": ["Soie", "Lin", "Neige", "Pierre", "Bois", "Ambre chaud"],
    },
    {
        "id": 8,
        "question": "Quelle plante aimez-vous le plus ?",
        "choices": ["Fleurs", "Plante aromatique", "Plante sauvage", "Patchiouli et vetiver", "Mousse sous bois", "Foin au soleil"],
    },
    {
        "id": 9,
        "question": "Quel type de parfum préférez-vous ?",
        "choices": ["Léger", "Caractère", "Vintage", "Chaud", "Frais", "Moderne"],
    },
    {
        "id": 10,
        "question": "Quelle matière première vous attire le plus ?",
        "choices": ["Musc", "Ambre", "Bois", "Gourmand", "Floral", "Marin"],
    },
    {
        "id": 11,
        "question": "Quelle couleurs préférez-vous ?",
        "choices": ["Marron et beige", "Vert", "Rouge", "Bleu", "Jaune", "Noir"],
    },
    {
        "id": 12,
        "question": "Quel type d'art vous attire le plus ?",
        "choices": ["Comprendre", "Reconnaitre", "Immaginer", "s'évader", "s'intérroger", "s'émmerveiller"],
    },
]
