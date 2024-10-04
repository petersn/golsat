import numpy as np
from PIL import Image
import autosat

frame_count = 4

img = Image.open('image.png').convert('L')
mask = np.array(img).T > 0
dims = mask.shape

inst = autosat.Instance()

@autosat.sat
def gol_constrain(
    x0, x1, x2,
    x3, x4, x5,
    x6, x7, x8,
):
    count = x0 + x1 + x2 + x3 + 0 + x5 + x6 + x7 + x8
    if count == 3:
        return 1
    if count == 2:
        return x4
    return 0

def make_grid():
    return [
        [inst.new_var() for _ in range(dims[1])]
        for _ in range(dims[0])
    ]

def constrain_grids(grid1, grid2):
    for i in range(dims[0]):
        for j in range(dims[1]):
            neighborhood = []
            for di in (-1, 0, +1):
                for dj in (-1, 0, +1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < dims[0] and 0 <= nj < dims[1]:
                        neighborhood.append(grid1[ni][nj])
                    else:
                        neighborhood.append(False)
            out = gol_constrain(*neighborhood)
            out.make_equal(grid2[i][j])

grids = [make_grid() for _ in range(frame_count)]
for i in range(frame_count - 1):
    constrain_grids(grids[i], grids[i + 1])

# Constrain the first frame to the mask
for i in range(dims[0]):
    for j in range(dims[1]):
        if mask[i, j]:
            grids[-1][i][j].make_equal(True)
        else:
            grids[-1][i][j].make_equal(False)

# Constrain the border to be dead in all frames
for t in range(frame_count):
    for i in range(dims[0]):
        grids[t][i][0].make_equal(False)
        grids[t][i][-1].make_equal(False)
    for j in range(dims[1]):
        grids[t][0][j].make_equal(False)
        grids[t][-1][j].make_equal(False)

with open('gol.dimacs', 'w') as f:
    f.write(inst.to_dimacs())

print("Solving...")
model = inst.solve(
    solver_name="glucose4",
    decode_model=False,
)
print("Solved.")

# Decode individual frames
for frame in range(frame_count):
    grid = grids[frame]
    img = Image.new('L', dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            img.putpixel((i, j), 255 if grid[i][j].decode(model) else 0)
    img.save(f'out/frame{frame}.png')
