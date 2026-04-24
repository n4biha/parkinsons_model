def fuse_predictions(clinical_prob, drawing_prob):
    # Check if models point in opposite directions
    clinical_says_pd = clinical_prob >= 0.5
    drawing_says_pd = drawing_prob >= 0.5
    
    # Average the probabilities
    avg = (clinical_prob + drawing_prob) / 2
    
    # True disagreement: models point opposite directions
    if clinical_says_pd != drawing_says_pd:
        return "Mixed Signals — models point in different directions, recommend professional evaluation"
    
    # Both agree, classify by average confidence
    if avg >= 0.65:
        return "High Risk — both indicators suggest PD"
    elif avg <= 0.35:
        return "Low Risk — both indicators suggest healthy"
    else:
        return "Moderate Risk — both indicators show some signs"