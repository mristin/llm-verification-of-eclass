"""
Experiment 4a: Synthetic similarity threshold.

Goal: Find the approximate distance threshold for duplicate detection.
Method:
1. define a 'Golden Set' of synonymous definitions (simulating typos, abbreviations).
2. Embed them using the SAME model used for the dataset.
3. Calculate the distance between them.
4. The Maximum Distance found becomes our baseline threshold.
"""

import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


model_name = "BAAI/bge-m3"

synthetic_pairs = [
    # Abbreviations (10 pairs)
    ("The maximum allowable temperature range for continuous industrial operation and handling",
     "The max allowable temperature range for continuous industrial operation and handling"),
    ("Standard specification defines the minimum required tensile strength for structural components",
     "Standard specification defines the min required tensile strength for structural components"),
    ("Manufacturing process documentation includes approximate dimensional tolerances for all critical features",
     "Manufacturing process documentation includes approx dimensional tolerances for all critical features"),
    ("Technical reference manual describes the operating temperature coefficient for precision measurement",
     "Technical reference manual describes the operating temp coefficient for precision measurement"),
    ("Equipment specification sheet lists the maximum permissible voltage for safe operation",
     "Equipment specification sheet lists the max permissible voltage for safe operation"),
    ("Installation guidelines recommend minimum clearance distances between adjacent electrical components",
     "Installation guidelines recommend min clearance distances between adjacent electrical components"),
    ("Product datasheet specifies the typical response time versus frequency characteristics",
     "Product datasheet specifies the typical response time vs frequency characteristics"),
    ("Quality control procedure requires verification of maximum deviation from nominal values",
     "Quality control procedure requires verification of max deviation from nominal values"),
    ("Safety documentation defines the minimum acceptable insulation resistance for compliance",
     "Safety documentation defines the min acceptable insulation resistance for compliance"),
    ("Calibration report indicates the approximate uncertainty margin for measurement accuracy",
     "Calibration report indicates the approx uncertainty margin for measurement accuracy"),

    # Punctuation (10 pairs)
    ("Primary input voltage rating for alternating current electrical systems and connections",
     "Primary input voltage rating (for alternating current electrical systems and connections)"),
    ("Operating frequency range measured in hertz for all standard configurations",
     "Operating frequency range, measured in hertz, for all standard configurations"),
    ("Output signal characteristics including amplitude phase and distortion parameters",
     "Output signal characteristics: amplitude, phase, and distortion parameters"),
    ("Protective housing material specification meets international standards for durability",
     "Protective housing material specification - meets international standards for durability"),
    ("Connection terminal arrangement follows industry standard wiring configuration and layout",
     "Connection terminal arrangement (follows industry standard wiring configuration and layout)"),
    ("Thermal dissipation capacity under maximum load conditions for extended operation",
     "Thermal dissipation capacity under maximum load conditions; for extended operation"),
    ("Mounting bracket dimensions specified in millimeters for universal compatibility",
     "Mounting bracket dimensions (specified in millimeters) for universal compatibility"),
    ("Electrical characteristics defined at reference ambient temperature and humidity conditions",
     "Electrical characteristics defined at reference ambient temperature and, humidity conditions"),
    ("Performance degradation curve shows relationship between operating time and efficiency",
     "Performance degradation curve shows relationship between: operating time and efficiency"),
    ("Input impedance value measured across entire operational bandwidth and frequency spectrum",
     "Input impedance value - measured across entire operational bandwidth and frequency spectrum"),

    # Casing (10 pairs)
    ("Nominal Operating Voltage Specification For Standard Industrial Equipment Applications",
     "nominal operating voltage specification for standard industrial equipment applications"),
    ("digital signal processing capability with advanced filtering and noise reduction",
     "Digital Signal Processing Capability With Advanced Filtering And Noise Reduction"),
    ("Maximum Continuous Current Rating Under Normal Ambient Temperature Conditions",
     "maximum continuous current rating under normal ambient temperature conditions"),
    ("environmental protection rating according to international ingress protection standards",
     "Environmental Protection Rating According To International Ingress Protection Standards"),
    ("Mechanical Stress Resistance Testing Protocol For Product Qualification And Validation",
     "mechanical stress resistance testing protocol for product qualification and validation"),
    ("output power capacity measured in watts for sustained operation",
     "Output Power Capacity Measured In Watts For Sustained Operation"),
    ("Installation Mounting Requirements Including Hardware Specifications And Torque Values",
     "installation mounting requirements including hardware specifications and torque values"),
    ("supply voltage tolerance range acceptable for reliable system operation",
     "Supply Voltage Tolerance Range Acceptable For Reliable System Operation"),
    ("Electromagnetic Interference Shielding Effectiveness Measured In Decibels Across Frequency Range",
     "electromagnetic interference shielding effectiveness measured in decibels across frequency range"),
    ("thermal cycling endurance specification for extended lifetime performance",
     "Thermal Cycling Endurance Specification For Extended Lifetime Performance"),

    # Spelling - localised (10 pairs)
    ("The outer protective casing color must match standard industrial equipment specifications",
     "The outer protective casing colour must match standard industrial equipment specifications"),
    ("High performance fiber optic transmission cable with enhanced signal integrity characteristics",
     "High performance fibre optic transmission cable with enhanced signal integrity characteristics"),
    ("Mechanical component requires proper lubrication and periodic maintenance during operation",
     "Mechanical component requires proper lubrication and periodic maintenance during operation"),
    ("Electronic circuit board utilizes advanced signal optimization techniques for performance",
     "Electronic circuit board utilises advanced signal optimization techniques for performance"),
    ("Precision analog measurement system designed for accurate data acquisition applications",
     "Precision analogue measurement system designed for accurate data acquisition applications"),
    ("Chemical vapor deposition process ensures uniform coating thickness across substrate",
     "Chemical vapour deposition process ensures uniform coating thickness across substrate"),
    ("Equipment must be properly grounded to ensure safe operation and protection",
     "Equipment must be properly earthed to ensure safe operation and protection"),
    ("Manufacturing catalog includes comprehensive listing of available product configurations and options",
     "Manufacturing catalogue includes comprehensive listing of available product configurations and options"),
    ("System requires proper initialization procedure before commencing normal operational mode",
     "System requires proper initialisation procedure before commencing normal operational mode"),
    ("Defense grade components meet stringent military specification requirements for reliability",
     "Defence grade components meet stringent military specification requirements for reliability"),

    # Word Order / Minor Phrasing (10 pairs)
    ("The nominal power consumption rating of the complete electrical assembly unit",
     "The complete electrical assembly unit nominal power consumption rating"),
    ("Recommended installation clearance distance for proper ventilation and thermal management",
     "Clearance distance for proper ventilation and thermal management recommended installation"),
    ("Standard operating temperature range specified for ambient environmental conditions",
     "Ambient environmental conditions specified standard operating temperature range"),
    ("Maximum permissible load capacity for structural mounting and support framework",
     "Structural mounting and support framework maximum permissible load capacity"),
    ("Typical response time characteristics measured under specified test conditions",
     "Measured under specified test conditions typical response time characteristics"),
    ("Input signal voltage level requirements for proper system functionality",
     "Proper system functionality input signal voltage level requirements"),
    ("External dimensions of the housing enclosure including mounting provisions",
     "Housing enclosure external dimensions including mounting provisions"),
    ("Minimum acceptable insulation resistance value for electrical safety compliance",
     "Electrical safety compliance minimum acceptable insulation resistance value"),
    ("Operating frequency bandwidth limitations for signal processing applications",
     "Signal processing applications operating frequency bandwidth limitations"),
    ("Net weight specification of the assembled unit without packaging materials",
     "Assembled unit net weight specification without packaging materials")
]


def get_logger():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger("Calibration")


if __name__ == "__main__":
    logger = get_logger()
    logger.info(f"Loading Model: {model_name}")

    model = SentenceTransformer(model_name)

    distances = []
    logger.info(f"\nCalculating Distances for {len(synthetic_pairs)} Pairs")

    for text_a, text_b in synthetic_pairs:
        embeddings = model.encode([text_a, text_b])
        dist = cosine_distances([embeddings[0]], [embeddings[1]])[0][0]
        distances.append(dist)
        logger.info(f"Dist: {dist:.5f} | '{text_a}' and '{text_b}'")


    max_dist = max(distances)
    avg_dist = sum(distances) / len(distances)

    # We recommend Max + small buffer to be safe
    rec_strict = max_dist
    rec_buffer = max_dist * 1.10

    logger.info("\n\nResults:")
    logger.info(f"Average Distance: {avg_dist:.5f}")
    logger.info(f"Maximum Distance: {max_dist:.5f} (The furthest synonym pair)")
    logger.info(f"Threshold (Strict): {rec_strict:.5f}")
    logger.info(f"Threshold (+10%):   {rec_buffer:.5f}")
    logger.info("Use the value above in the next script (4b_run_discovery.py)")