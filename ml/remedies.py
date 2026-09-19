"""Treatment guidance returned alongside each prediction."""

REMEDIES: dict[str, str] = {
    "default": "Remove affected leaves, improve air circulation and avoid overhead "
               "watering. Consult a local agronomist before applying chemicals.",
    "Tomato___Late_blight": "Destroy infected plants, avoid overhead irrigation and "
                            "spray chlorothalonil or mancozeb every 7-10 days.",
    "Tomato___Early_blight": "Rotate crops, mulch the soil and apply a copper-based "
                             "fungicide at first symptoms.",
    "Tomato___healthy": "No disease detected. Maintain balanced NPK feeding and "
                        "regular scouting.",
    "Potato___Late_blight": "Use certified seed tubers, hill the soil and apply "
                            "metalaxyl-based fungicide preventively.",
    "Potato___Early_blight": "Ensure adequate nitrogen, remove debris and spray "
                             "azoxystrobin on a 10-day schedule.",
    "Corn_(maize)___Common_rust_": "Plant resistant hybrids; apply a triazole "
                                   "fungicide if pustules cover over 5% of leaf area.",
    "Apple___Apple_scab": "Rake and destroy fallen leaves; apply captan or myclobutanil "
                          "from green tip through petal fall.",
    "Grape___Black_rot": "Prune out mummified fruit and apply mancozeb from bud break.",
    "Pepper,_bell___Bacterial_spot": "Use copper plus mancozeb sprays and pathogen-free "
                                     "seed; avoid working plants when wet.",
}
