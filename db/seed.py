"""Database seeding script populating Indian Cattle and Buffalo breed metadata and initial model version."""

import logging
from sqlalchemy.orm import Session

from db.session import engine, init_db, SessionLocal
from db.repository import BreedRepository, ModelVersionRepository
from db.models import Breed, ModelVersion

logger = logging.getLogger("db_seed")

INITIAL_BREEDS = [
    {
        "breed_name": "Gir",
        "animal_type": "cattle",
        "origin": "Gir Hills & Kathiawar Forests",
        "native_state": "Gujarat",
        "physical_characteristics": {
            "forehead": "Prominent, broad, and convex rounded forehead shield",
            "horns": "Peculairly curved backward and downward, turning upward at tips",
            "coat_color": "Varies from red, speckled yellow to white and reddish black",
            "ears": "Long, pendulous, folded like a leaf with a notch at tip",
            "hump": "Large, prominent muscular hump",
            "body_frame": "Medium to large well-proportioned body frame",
        },
        "milk_production": {
            "average_lactation_yield_kg": 2182,
            "fat_percentage_range": "4.5% - 5.0%",
            "peak_daily_yield_kg": 15.5,
        },
        "climate_adaptability": "Highly heat-tolerant, disease-resistant to tropical ectoparasites, thrives under harsh arid and semi-arid tropical climates.",
        "uses": "dairy",
        "description": "Gir is one of the premier zebu dairy cattle breeds of India originating from the Gir forest region. Renowned globally for high milk yield, gentle temperament, and A2 milk protein composition.",
    },
    {
        "breed_name": "Ongole",
        "animal_type": "cattle",
        "origin": "Ongole Taluk, Prakasam District",
        "native_state": "Andhra Pradesh",
        "physical_characteristics": {
            "forehead": "Broad, flat or slightly convex between eyes",
            "horns": "Short, stumpy, growing backward and outward",
            "coat_color": "Glossy white in females; dark grey markings on hump and neck in males",
            "ears": "Moderately long, drooping slightly",
            "hump": "Erect, muscular, and well-developed hump",
            "body_frame": "Heavy, muscular, large draft body frame with majestic stance",
        },
        "milk_production": {
            "average_lactation_yield_kg": 1500,
            "fat_percentage_range": "4.2% - 4.8%",
            "peak_daily_yield_kg": 10.0,
        },
        "climate_adaptability": "Exceptional heat tolerance, tropical disease immunity, and capability to perform heavy agricultural draft work under high temperatures.",
        "uses": "dual-purpose",
        "description": "Ongole is a world-famous dual-purpose draft and dairy zebu cattle breed native to Andhra Pradesh. Widely exported globally for beef and draft crossbreeding due to immense strength and tick resistance.",
    },
    {
        "breed_name": "Sahiwal",
        "animal_type": "cattle",
        "origin": "Sahiwal / Montgomery Region",
        "native_state": "Punjab",
        "physical_characteristics": {
            "forehead": "Medium broad, flat",
            "horns": "Short and thick, stumpy",
            "coat_color": "Reddish brown to pale red/dun coat",
            "ears": "Medium sized, drooping",
            "hump": "Well-developed massive hump in bulls",
            "body_frame": "Heavy built, loose skin (Zebu 'Lola'), short legs",
        },
        "milk_production": {
            "average_lactation_yield_kg": 2325,
            "fat_percentage_range": "4.8% - 5.2%",
            "peak_daily_yield_kg": 18.0,
        },
        "climate_adaptability": "High tolerance to extreme heat and tick infestations, excellent feed conversion efficiency under tropical conditions.",
        "uses": "dairy",
        "description": "Sahiwal is regarded as the best indigenous dairy cattle breed of the Indian subcontinent. Known for docility, rapid milk let-down, and rich fat-content dairy production.",
    },
    {
        "breed_name": "Jaffarabadi",
        "animal_type": "buffalo",
        "origin": "Gir Forest & Coastal Saurashtra",
        "native_state": "Gujarat",
        "physical_characteristics": {
            "forehead": "Prominently ultra-broad, dome-shaped convex forehead",
            "horns": "Heavy, broad, flat horns drooping down along cheeks then curling upward",
            "coat_color": "Usually jet black",
            "ears": "Broad and drooping",
            "hump": "None (Riverine Buffalo)",
            "body_frame": "Massive, heavy, deep body frame — the largest of Indian buffalo breeds",
        },
        "milk_production": {
            "average_lactation_yield_kg": 2150,
            "fat_percentage_range": "7.5% - 8.5%",
            "peak_daily_yield_kg": 16.0,
        },
        "climate_adaptability": "Well-adapted to hot coastal and marshy forest environments, excellent wallowing and foraging capability.",
        "uses": "dairy",
        "description": "Jaffarabadi is the heaviest riverine buffalo breed of India, native to the Gir forest region. Famed for exceptionally high butterfat milk ideal for ghee production.",
    },
    {
        "breed_name": "Murrah",
        "animal_type": "buffalo",
        "origin": "Rohtak, Hisar, and Jind Districts",
        "native_state": "Haryana",
        "physical_characteristics": {
            "forehead": "Slightly convex, broad",
            "horns": "Short, tightly curled in spiral fashion close to head",
            "coat_color": "Jet black with occasional white switch on tail",
            "ears": "Short, thin, and alert",
            "hump": "None (Riverine Buffalo)",
            "body_frame": "Deep wedge-shaped body frame with well-developed udder and milk veins",
        },
        "milk_production": {
            "average_lactation_yield_kg": 2500,
            "fat_percentage_range": "7.0% - 7.8%",
            "peak_daily_yield_kg": 20.0,
        },
        "climate_adaptability": "Highly adaptable across diverse agro-climatic zones nationwide; responds exceptionally well to commercial dairy management.",
        "uses": "dairy",
        "description": "Murrah is the undisputed premier dairy buffalo breed of the world, nicknamed 'Black Gold'. It forms the backbone of commercial milk production in northern India.",
    },
    {
        "breed_name": "Surti",
        "animal_type": "buffalo",
        "origin": "Kheda & Vadodara Region",
        "native_state": "Gujarat",
        "physical_characteristics": {
            "forehead": "Broad and flat between horns",
            "horns": "Sickle-shaped, flat, growing backward and downward before curving up",
            "coat_color": "Black or rusty brown with two distinct white collars on brisket/neck",
            "ears": "Medium sized with reddish hair inside",
            "hump": "None (Riverine Buffalo)",
            "body_frame": "Medium-sized, compact, well-built straight body frame",
        },
        "milk_production": {
            "average_lactation_yield_kg": 1650,
            "fat_percentage_range": "7.5% - 8.2%",
            "peak_daily_yield_kg": 12.0,
        },
        "climate_adaptability": "Thrives under semi-arid conditions with moderate feeding, highly economical feed-to-milk conversion ratio.",
        "uses": "dairy",
        "description": "Surti is an efficient medium-sized riverine buffalo breed native to Gujarat. Popular among smallholder farmers for compact size, low maintenance cost, and high fat milk.",
    },
]


def seed_database(db: Session) -> None:
    """Seed initial breed metadata catalog and active model version."""
    logger.info("Starting database seeding process...")

    # 1. Seed Breed Metadata
    seeded_breeds_count = 0
    for bdata in INITIAL_BREEDS:
        existing = BreedRepository.get_by_name(db, bdata["breed_name"])
        if not existing:
            BreedRepository.create_breed(
                db=db,
                breed_name=bdata["breed_name"],
                animal_type=bdata["animal_type"],
                origin=bdata["origin"],
                native_state=bdata["native_state"],
                physical_characteristics=bdata["physical_characteristics"],
                milk_production=bdata["milk_production"],
                climate_adaptability=bdata["climate_adaptability"],
                uses=bdata["uses"],
                description=bdata["description"],
            )
            seeded_breeds_count += 1
            logger.info("Seeded breed: %s (%s)", bdata["breed_name"], bdata["animal_type"])

    # 2. Seed Default Active Model Version
    active_version = ModelVersionRepository.get_active(db)
    if not active_version:
        ModelVersionRepository.create_version(
            db=db,
            model_name="Indian Cattle & Buffalo Breed Recognition System",
            yolo_version="YOLOv8n-AnimalDetection-v1.0",
            efficientnet_version="EfficientNet-B0-BreedRecognition-v1.0",
            gradcam_version="Grad-CAM-v1.0",
            is_active=True,
        )
        logger.info("Seeded initial active ModelVersion record.")

    logger.info(
        "Seeding process complete. New breeds added: %d, Total breeds: %d",
        seeded_breeds_count,
        len(BreedRepository.list_all(db)),
    )


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
