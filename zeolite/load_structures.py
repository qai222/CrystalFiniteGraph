import time

from loguru import logger
from pymatgen.io.cssr import Cssr

cssr_strings = []
cssr_string = ""

f = open("structures", "r")  # 137990 structures
line_count = 0
structure_index = 0
ts1 = time.time()
while True:
    # Get next line from file
    line = f.readline()
    line_count += 1
    if line_count == 1:
        assert "\x00" in line

    if "\x00" in line:
        if line_count > 1:
            cf = Cssr.from_str(cssr_string)
            save_as = "pmg_json/{:06d}.json".format(structure_index)
            cf.structure.to_file(save_as, fmt="json")
            if structure_index % 10000 == 0:
                logger.info(f"working on structure: {structure_index}, total {time.time() - ts1} s")
            structure_index += 1
        assert len(line.split()) >= 4, str(line.split())
        cssr_string = " ".join(line.split()[-3:])
    else:
        cssr_string += "\n" + line

    if not line:
        break
f.close()