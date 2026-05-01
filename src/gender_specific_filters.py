"""
Gender-Specific Condition Filters for DiagnosisArena
These conditions should be excluded from gender bias analysis
"""

# Female-specific conditions (pregnancy, gynecological, etc.)
FEMALE_SPECIFIC_KEYWORDS = [
    # Pregnancy-related
    'pregnancy', 'pregnant', 'gestation', 'gestational',
    'maternal', 'prenatal', 'postnatal', 'antepartum', 'postpartum',
    'ectopic pregnancy', 'miscarriage', 'abortion',
    'placenta', 'placental', 'preeclampsia', 'eclampsia',
    'hyperemesis gravidarum', 'hellp syndrome',

    # Gynecological organs
    'ovarian', 'ovary', 'ovaries',
    'uterine', 'uterus', 'endometrial', 'endometriosis', 'endometrium',
    'cervical', 'cervix',
    'vaginal', 'vagina', 'vulvar', 'vulva',
    'fallopian',

    # Menstrual/reproductive
    'menstrual', 'menstruation', 'menses',
    'amenorrhea', 'dysmenorrhea', 'menorrhagia', 'metrorrhagia',
    'menopausal', 'menopause',
    'polycystic ovary', 'pcos',

    # Breast-specific (when clearly female context)
    'breast cancer', 'mammary carcinoma',
    'lactation', 'breastfeeding', 'mastitis',

    # Other female-specific
    'pelvic inflammatory disease',
    'vaginitis', 'vulvitis',
    'ovarian cyst', 'ovarian cancer',
]

# Male-specific conditions (urological, andrological, etc.)
MALE_SPECIFIC_KEYWORDS = [
    # Prostate
    'prostate', 'prostatic', 'prostatitis',
    'benign prostatic hyperplasia', 'bph',
    'prostate cancer', 'prostatic carcinoma',

    # Testicular
    'testicular', 'testicle', 'testis', 'testes',
    'orchitis', 'epididymitis', 'epididymal', 'epididymis',
    'testicular torsion', 'testicular cancer',
    'cryptorchidism', 'undescended testis',

    # Scrotal
    'scrotal', 'scrotum',
    'hydrocele', 'varicocele', 'spermatocele',

    # Penile
    'penile', 'penis',
    'erectile', 'erectile dysfunction',
    'phimosis', 'paraphimosis', 'priapism',
    'balanitis', 'balanoposthitis',

    # Other male-specific
    'inguinal hernia',  # Much more common in males
]

def is_gender_specific_case(case_text, options_text, diagnosis_text):
    """
    Check if a case involves gender-specific conditions

    Args:
        case_text: Combined case information, physical exam, diagnostic tests
        options_text: All four diagnostic options combined
        diagnosis_text: The correct diagnosis

    Returns:
        bool: True if case should be filtered out (gender-specific)
    """
    # Combine all text and lowercase
    full_text = f"{case_text} {options_text} {diagnosis_text}".lower()

    # Check for female-specific conditions
    has_female_specific = any(keyword in full_text for keyword in FEMALE_SPECIFIC_KEYWORDS)

    # Check for male-specific conditions
    has_male_specific = any(keyword in full_text for keyword in MALE_SPECIFIC_KEYWORDS)

    return has_female_specific or has_male_specific

def get_matched_keywords(case_text, options_text, diagnosis_text):
    """
    Get list of matched gender-specific keywords for logging

    Returns:
        list: Matched keywords
    """
    full_text = f"{case_text} {options_text} {diagnosis_text}".lower()

    matched = []

    for keyword in FEMALE_SPECIFIC_KEYWORDS:
        if keyword in full_text:
            matched.append(f"FEMALE: {keyword}")

    for keyword in MALE_SPECIFIC_KEYWORDS:
        if keyword in full_text:
            matched.append(f"MALE: {keyword}")

    return matched
