

def fuse_predictions(clinical_prob, drawing_prob):
    diff = abs(clinical_prob - drawing_prob)

    if clinical_prob >= 0.65 and drawing_prob >= 0.65:
        return "High Risk — both models agree"
    elif clinical_prob <= 0.35 and drawing_prob <= 0.35:
        return "Low Risk — both models agree"
    elif diff > 0.3:
        return "Inconclusive — models disagree"
    else:
        return "Moderate Risk"
