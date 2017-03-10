import os
import math
import random
import time
import shutil
import itertools
import multiprocessing as mp
from functools import partial

import numpy as np
import cv2


random.seed(int(time.time()))


# *****************</MISC>*****************
def imwrite(name, img):
    ext = os.path.splitext(name)[-1].lower()
    params = None
    if any(ext == jex for jex in ['jpeg', 'jpg', 'jpe', 'jp2']):
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext == 'png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]

    return cv2.imwrite(name, img, params)


def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def make_output_dirs(src, output_dir):
    src = os.path.relpath(src)
    classes = get_dirlist_recursively(src)

    for class_dir in classes:
        ocd = os.path.join(output_dir, class_dir[len(src)+1:])
        if not os.path.exists(ocd):
            os.makedirs(ocd)


def timeit(method):
    stars = '*'*79
    def timer(*args, **kwargs):
        print('\n'+stars)
        print('{name} started'.format(name=method.__name__))

        start_time = time.time()
        result = method(*args, **kwargs)

        print(
            '{name} finished for {time:.4f} seconds'.format(
                name=method.__name__,
                time=(time.time() - start_time)
            )
        )
        print(stars)

        return result

    return timer


def listdir_fullpath(dirname):
    return [os.path.join(dirname, f) for f in os.listdir(dirname)]


def get_dirlist_recursively(dirname):
    return [root for root, d, f in os.walk(dirname)]


def get_cos_sin(x):
    x = math.radians(x)
    return math.cos(x), math.sin(x)
# *****************</MISC>*****************


def get_M(x, y, z, size):
    w, h = size
    half_w = w*.5
    half_h = h*.5
    center_x = half_w
    center_y = half_h

    cos_x, sin_x = get_cos_sin(x)
    cos_y, sin_y = get_cos_sin(y)
    cos_z, sin_z = get_cos_sin(z)

    # Rotation matrix:
    # | cos(y)*cos(z)                       -cos(y)*sin(z)                     sin(y)         0 |
    # | cos(x)*sin(z)+cos(z)*sin(x)*sin(y)  cos(x)*cos(z)-sin(x)*sin(y)*sin(z) -cos(y)*sin(y) 0 |
    # | sin(x)*sin(z)-cos(x)*sin(y)*sin(z)  sin(x)*sin(z)+cos(x)*sin(y)*sin(z) cos(x)*cos(y)  0 |
    # | 0                                   0                                  0              1 |

    R = np.float32(
        [
            [cos_y * cos_z,  cos_x * sin_z + cos_z * sin_y * sin_x],
            [-cos_y * sin_z, cos_z * cos_x - sin_z * sin_y * sin_x],
            [sin_y,          cos_y * sin_x],
        ]
    )

    center = np.float32([center_x, center_y])
    offset = np.float32(
        [
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ]
    )

    points_z = np.dot(offset, R[2])
    dev_z = np.vstack([w/(w + points_z), h/(h + points_z)])

    new_points = np.dot(offset, R[:2].T) * dev_z.T + center
    in_pt = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    transform = cv2.getPerspectiveTransform(in_pt, new_points)
    return transform


def get_random_M(size):
    # TODO: set correct values
    x = uniform(-10, 10)
    y = uniform(-10, 10)
    z = uniform(-10, 10)

    return get_M(x, y, z, size)


def transform_img(img, theta=None, phi=None, gamma=None):
    size = img.shape[:2]

    if theta is None or phi is None or gamma is None:
        M = get_random_M(size)
    else:
        M = get_M(theta, phi, gamma, size)

    result = cv2.warpPerspective(img, M, size)

    return result


def get_angles(count):
    #     |y
    #_____|_____x
    #     |
    #     |
    # z - clockwise rotations
    edge = 26
    x = {'min':   0, 'max': 30, '0': 0}
    y = {'min': -30, 'max': 30, '0': 0}
    z = {'min':   5, 'max': -5, '0': 0}

    def get_rx(): return random.uniform(x['min'], x['max'])
    def get_ry(): return random.uniform(y['min'], y['max'])
    def get_rz(): return random.uniform(z['max'], z['min'])

    perm = itertools.product(('min', 'max', '0'), repeat=3)

    angles_set = [(x[xk], y[yk], z[zk]) for xk, yk, zk in perm
                                    if (xk, yk, zk) != (0, 0, 0)]

    if count > edge:
        diff = count - edge
        for i in range(diff):
            new_angles = (get_rx(), get_ry(), get_rz())
            angles_set.append(new_angles)

    tmp = []

    for angles_el in angles_set:
        x, y, z = angles_el
        w = 0
        if y != 0:
            w = 0.6
        elif x != 0:
            w = 0.3
        else:
            w = 0.1
        tmp.append((angles_el, w))

    angles_set = tmp[:]
    del tmp

    for i in range(count):
        choice = weighted_choice(angles_set)

        yield choice


def rotate_images_in_dir(class_dir, output_dir, goal):
    process_name = '[{}]'.format(mp.current_process().name)
    log_string = '{process_name} {dir} finished. Total saved files count {{total}}'.format(process_name=process_name, dir=os.path.basename(class_dir))

    files = listdir_fullpath(class_dir)
    output_dir = os.path.join(output_dir, os.path.basename(class_dir))

    total = 0
    files_count = len(files)

    if files_count == 0:
        print(log_string.format(total=total))
        return total

    if files_count > goal:
        files = random.sample(files, goal / 2)
        files_count = len(files)

    rotations_count = int(math.ceil(float(goal) / files_count))

    for fname in files:
        if total >= goal:
            break
        if total + rotations_count + 1 > goal:
            rotations_count = max(goal - total - 1, 0)

        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        imname = os.path.basename(fname)

        for idx, angles_el in enumerate(get_angles(rotations_count)):
            img_r = transform_img(img, *angles_el)
            nn = '{0[0]}_{n}{0[1]}'.format(os.path.splitext(imname), n=idx)
            oimpath = os.path.join(output_dir, nn)
            imwrite(oimpath, img_r)

        shutil.copy(fname, output_dir)
        total += rotations_count + 1

    print(log_string.format(total=total))

    return total


@timeit
def augment_dataset(dataset_dir, output_dir=None, goal=2000, processes=4):
    dataset_dir = os.path.relpath(dataset_dir)
    if not output_dir:
        output_dir = dataset_dir
    print('source dir: {}\noutput dir: {}\n'.format(dataset_dir, output_dir))

    classes = listdir_fullpath(dataset_dir)
    print('classes: {}'.format(
        ', '.join(os.path.basename(c) for c in classes))
    )

    # for every class different process
    foo = partial(rotate_images_in_dir, output_dir=output_dir, goal=goal)
    pool = mp.Pool(processes=processes)

    res = pool.map_async(foo, classes)

    pool.close()
    pool.join()

    print('Total processed {}'.format(sum(res.get())))


if __name__ == '__main__':
    import sys
    inp = sys.argv[1]
    out = sys.argv[2]
    try:
        goal = sys.argv[3]
    except:
        goal = 100
    make_output_dirs(inp, out)
    augment_dataset(inp, out, goal)
