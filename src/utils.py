import cv2
import numpy as np
from pathlib import Path
import json
import glob
import os

def get_contour_feats(cnt):
    """Get morphological features from input contours."""
    cnt = np.array(cnt)
    # calculate some things useful later:
    m = cv2.moments(cnt)

    # ** regionprops **
    Area = m["m00"]
   
    Perimeter = cv2.arcLength(cnt, True)
    # bounding box: x,y,width,height
    BoundingBox = cv2.boundingRect(cnt)
    # centroid    = m10/m00, m01/m00 (x,y)
    Centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])

    # EquivDiameter: diameter of circle with same area as region
    EquivDiameter = np.sqrt(4 * Area / np.pi)
    # Extent: ratio of area of region to area of bounding box
    Extent = Area / (BoundingBox[2] * BoundingBox[3])

    # CONVEX HULL 
    # convex hull vertices
    ConvexHull = cv2.convexHull(cnt)
    ConvexArea = cv2.contourArea(ConvexHull)
    # Solidity := Area/ConvexArea
    Solidity = Area / ConvexArea
    #there should be at least 5 points to fit the ellipse in function 'fitEllipseNoDirect'
    """# ELLIPSE - determine best-fitting ellipse.
    centre, axes, angle = cv2.fitEllipse(cnt)
    MAJ = np.argmax(axes)  # this is MAJor axis, 1 or 0
    MIN = 1 - MAJ  # 0 or 1, MINor axis
    # Note: axes length is 2*radius in that dimension
    MajorAxisLength = axes[MAJ]
    MinorAxisLength = axes[MIN]
    Eccentricity = np.sqrt(1 - (axes[MIN] / axes[MAJ]) ** 2)
    Orientation = angle
    EllipseCentre = centre  # x,y"""
    return {
        "area": Area,
        "perimeter": Perimeter,
        "equiv-diameter": EquivDiameter,
        "extent": Extent,
        "convex-area": ConvexArea,
        "solidity": Solidity
    }

    return {
        "area": Area,
        "perimeter": Perimeter,
        "equiv-diameter": EquivDiameter,
        "extent": Extent,
        "convex-area": ConvexArea,
        "solidity": Solidity,
        "major-axis-length": MajorAxisLength,
        "minor-axis-length": MinorAxisLength,
        "eccentricity": Eccentricity,
        "orientation": Orientation,
        "ellipse-centre-x": EllipseCentre[0],
        "ellipse-centre-y": EllipseCentre[1],
    }

def directory_samples(directory):
    files = glob.glob(os.path.join(directory, '*.json'))
    # Get the count of JSON files
    count = len(files)
    return count

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return round(self.avg, 8)