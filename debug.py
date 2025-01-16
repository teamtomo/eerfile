import time

from eerfile.render_eer import render
from tifffile import TiffFile

file = "/Users/burta2/data/FoilHole_19622436_Data_19616947_19616949_20241218_121657_EER.eer"

start_time = time.time()

with TiffFile(file) as tiff:
    image = tiff.asarray()

end_time = time.time()
print(f"Execution Time: {end_time - start_time:.6f} seconds")


start_time = time.time()
# Code block to time
image = render(
    file=file,
    dose_per_output_frame=1,
)

end_time = time.time()

print(f"Execution Time: {end_time - start_time:.6f} seconds")



import napari
viewer = napari.Viewer()
viewer.add_image(image)
napari.run()