import zarr

# need to register, otherwise issues come up: https://github.com/cgohlke/imagecodecs/issues/82
import numcodecs 
from imagecodecs.numcodecs import Jpegxl 
numcodecs.register_codec(Jpegxl) 

store = zarr.open("task2_tmi_sanity_check.zarr.zip")

print(list(store))

for column in list(store["data"]):
    print(store["data"][column])

for i in range(1):
    for column in list(store["data"]):
        print(store["data"][column][i])
