from pathlib import Path
from heatmap_line_extraction import HeatmapExtractor

base_dir = Path(__file__).parent

image_file = base_dir / "example_heatmap.png"

extractor = HeatmapExtractor(
    image_file=image_file,
    resolution=(300, 300),
    value_range=(0, 100),
    x_range=(0, 75),
    y_range=(0, 90),
    debug=True
)

extractor.extract_profile(
    profile_type="vertical",
    x=40
)
