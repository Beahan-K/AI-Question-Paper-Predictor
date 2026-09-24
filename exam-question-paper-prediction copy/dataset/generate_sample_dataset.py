"""
generate_sample_dataset.py
---------------------------
One-time helper script that builds a SAMPLE prototype dataset of
~360 NEET-style questions spread across 2 papers (2019, 2022) and
3 subjects (Physics, Chemistry, Biology).

This is prototype/demo data only, meant to let the application run
end-to-end out of the box. Replace dataset/questions.csv with real
question-paper data (or use the "Upload Question Paper" page) at
any time -- the rest of the application does not need to change,
because it only depends on the generic column names:

    ID, Year, Subject, Question_No, Chapter, Question, Topic, Marks

Run:
    python dataset/generate_sample_dataset.py
"""

import csv
import random
import os

random.seed(42)

# Chapter -> Topic -> list of question stems, per subject.
SUBJECT_CHAPTERS = {
    "Physics": {
        "Mechanics": ["Laws of Motion", "Work Energy Power", "Rotational Motion", "Gravitation"],
        "Thermodynamics": ["Heat Transfer", "Laws of Thermodynamics", "Kinetic Theory of Gases"],
        "Electrostatics": ["Coulomb's Law", "Electric Field", "Capacitance", "Gauss's Law"],
        "Current Electricity": ["Ohm's Law", "Kirchhoff's Laws", "Resistivity", "Cells and EMF"],
        "Magnetism": ["Magnetic Field", "Electromagnetic Induction", "AC Circuits"],
        "Optics": ["Reflection and Refraction", "Wave Optics", "Optical Instruments"],
        "Modern Physics": ["Photoelectric Effect", "Atomic Structure", "Nuclear Physics", "Semiconductors"],
        "Oscillations and Waves": ["Simple Harmonic Motion", "Sound Waves", "Doppler Effect"],
    },
    "Chemistry": {
        "Organic Chemistry": ["Hydrocarbons", "Alcohols Phenols Ethers", "Aldehydes and Ketones", "Amines", "Biomolecules"],
        "Inorganic Chemistry": ["Periodic Table", "Chemical Bonding", "Coordination Compounds", "p-Block Elements", "d and f Block Elements"],
        "Physical Chemistry": ["Chemical Kinetics", "Thermodynamics", "Equilibrium", "Electrochemistry", "Solutions"],
        "Atomic Structure": ["Quantum Numbers", "Electronic Configuration", "Bohr Model"],
        "Chemical Bonding": ["Ionic Bonding", "Covalent Bonding", "Hybridization", "VSEPR Theory"],
        "Environmental Chemistry": ["Pollution", "Green Chemistry"],
    },
    "Biology": {
        "Genetics": ["Mendelian Inheritance", "DNA Replication", "Gene Expression", "Chromosomal Disorders"],
        "Human Physiology": ["Digestive System", "Circulatory System", "Nervous System", "Excretory System", "Respiratory System"],
        "Plant Physiology": ["Photosynthesis", "Respiration in Plants", "Plant Growth and Hormones"],
        "Ecology": ["Ecosystem", "Biodiversity", "Population Ecology", "Environmental Issues"],
        "Cell Biology": ["Cell Structure", "Cell Division", "Cell Organelles"],
        "Reproduction": ["Human Reproduction", "Plant Reproduction", "Reproductive Health"],
        "Evolution": ["Natural Selection", "Origin of Life", "Evidence of Evolution"],
        "Biotechnology": ["Genetic Engineering", "PCR", "Recombinant DNA Technology"],
    },
}

QUESTION_TEMPLATES = [
    "Which of the following statements about {topic} is correct in the context of {chapter}?",
    "Explain the concept of {topic} with a suitable example from {chapter}.",
    "A {subject} question based on {topic} was asked regarding its practical application in {qualifier}.",
    "Identify the correct option related to {topic} from the given choices ({qualifier}).",
    "What is the significance of {topic} in the chapter on {chapter} for {qualifier}?",
    "Calculate the value based on the principle of {topic}, given data on {qualifier}.",
    "Describe the process involved in {topic} as covered under {chapter}.",
    "Which factor does NOT affect {topic} during {qualifier}?",
    "State the law/principle associated with {topic} and its role in {qualifier}.",
    "Compare and contrast two aspects of {topic} with reference to {qualifier}.",
    "The diagram represents {topic} in the context of {chapter}. Identify the labelled parts.",
    "Which of the following best explains the mechanism of {topic} observed during {qualifier}?",
]

QUALIFIERS = [
    "laboratory conditions", "standard temperature and pressure", "real-world systems",
    "the given experiment", "a typical exam scenario", "recent research findings",
    "textbook case studies", "field observations", "controlled experiments",
    "industrial applications", "biological systems", "everyday phenomena",
]

QUESTION_TYPES = ["MCQ", "Assertion-Reason", "Numerical", "Conceptual"]

YEARS = [2019, 2022]
MARKS_OPTIONS = [4, 4, 4, 5]  # NEET is mostly 4 marks per question

rows = []
qid = 1

TARGET_TOTAL = 360
subjects = list(SUBJECT_CHAPTERS.keys())

# Build a flat pool of (subject, chapter, topic) combos
pool = []
for subject, chapters in SUBJECT_CHAPTERS.items():
    for chapter, topics in chapters.items():
        for topic in topics:
            pool.append((subject, chapter, topic))

# We want ~360 questions split across 2 years, roughly evenly across subjects
per_year = TARGET_TOTAL // len(YEARS)

for year in YEARS:
    # give more weight to certain topics to create realistic "frequently repeated" topics
    weighted_pool = pool + random.sample(pool, k=len(pool) // 2)
    random.shuffle(weighted_pool)

    q_no_per_subject = {s: 1 for s in subjects}

    for i in range(per_year):
        subject, chapter, topic = weighted_pool[i % len(weighted_pool)]
        template = random.choice(QUESTION_TEMPLATES)
        qualifier = random.choice(QUALIFIERS)
        question_text = template.format(
            topic=topic.lower(), chapter=chapter.lower(),
            subject=subject, qualifier=qualifier,
        )
        marks = random.choice(MARKS_OPTIONS)
        q_no = q_no_per_subject[subject]
        q_no_per_subject[subject] += 1

        rows.append({
            "ID": qid,
            "Year": year,
            "Subject": subject,
            "Question_No": q_no,
            "Chapter": chapter,
            "Question": question_text,
            "Topic": topic,
            "Marks": marks,
        })
        qid += 1

# Ensure exactly TARGET_TOTAL rows (trim/pad if rounding caused a mismatch)
rows = rows[:TARGET_TOTAL]

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "questions.csv")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ID", "Year", "Subject", "Question_No", "Chapter", "Question", "Topic", "Marks"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} sample questions -> {out_path}")
